/* QNN 2.37 HTP-only, persistent context. No CPU provider or file IO per frame. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpCommon.h"
typedef Qnn_ErrorHandle_t (*QnnInterface_GetProvidersFn_t)(const QnnInterface_t ***, uint32_t *);
typedef Qnn_ErrorHandle_t (*QnnSystemInterface_GetProvidersFn_t)(const QnnSystemInterface_t ***, uint32_t *);

#define FIELD(t, f) ((t).version == QNN_TENSOR_VERSION_1 ? (t).v1.f : (t).v2.f)
typedef struct {
  void *htp_lib, *system_lib, *stub_lib;
  QNN_INTERFACE_VER_TYPE api;
  QNN_SYSTEM_INTERFACE_VER_TYPE sys;
  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device;
  Qnn_ContextHandle_t context;
  Qnn_GraphHandle_t graph;
  QnnSystemContext_Handle_t metadata;
  Qnn_LogHandle_t logger;
  void *binary;
  Qnn_Tensor_t input, outputs[4];
  unsigned char *buffers[5];
  unsigned output_index[4];
  char error[512];
} Session;

static void log_callback(const char *fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
  (void)level; (void)ts;
  fputs("[QNN HTP] ", stderr); vfprintf(stderr, fmt, args); fputc('\n', stderr);
}
static int fail(Session *s, const char *step, Qnn_ErrorHandle_t code) {
  snprintf(s->error, sizeof(s->error), "%s failed: 0x%llx", step, (unsigned long long)code);
  return -1;
}
static void *open_library(const char *directory, const char *name) {
  char path[4096];
  if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= (int)sizeof(path)) return NULL;
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
void q6a_destroy(void *opaque) {
  Session *s = opaque; if (!s) return;
  if (s->context) s->api.contextFree(s->context, NULL);
  if (s->device) s->api.deviceFree(s->device);
  if (s->backend) s->api.backendFree(s->backend);
  if (s->logger) s->api.logFree(s->logger);
  for (unsigned i=0;i<5;i++) free(s->buffers[i]);
  if (s->metadata) s->sys.systemContextFree(s->metadata);
  free(s->binary);
  if (s->htp_lib) dlclose(s->htp_lib);
  if (s->system_lib) dlclose(s->system_lib);
  if (s->stub_lib) dlclose(s->stub_lib);
  free(s);
}
static int bind_tensor(Session *s, Qnn_Tensor_t *t, unsigned slot,
                       const char *name, const uint32_t *dims, uint32_t rank) {
  if (t->version != QNN_TENSOR_VERSION_1 && t->version != QNN_TENSOR_VERSION_2)
    return fail(s,"tensor version",t->version);
  if (!FIELD(*t,name) || strcmp(FIELD(*t,name),name) || FIELD(*t,rank)!=rank ||
      FIELD(*t,dataType)!=QNN_DATATYPE_UFIXED_POINT_8 ||
      FIELD(*t,dataFormat)!=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER ||
      FIELD(*t,quantizeParams).quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    return fail(s,"unexpected tensor metadata",slot);
  uint32_t size=1;
  for (uint32_t i=0;i<rank;i++) {
    if (FIELD(*t,dimensions)[i]!=dims[i]) return fail(s,"tensor dimensions",slot);
    size*=dims[i];
  }
  float scale=FIELD(*t,quantizeParams).scaleOffsetEncoding.scale;
  if (!isfinite(scale) || scale<=0) return fail(s,"quantization scale",slot);
  s->buffers[slot]=calloc(size,1);
  if (!s->buffers[slot]) return fail(s,"tensor allocation",slot);
  if (t->version==QNN_TENSOR_VERSION_1) {
    t->v1.memType=QNN_TENSORMEMTYPE_RAW; t->v1.clientBuf.data=s->buffers[slot]; t->v1.clientBuf.dataSize=size;
  } else {
    t->v2.memType=QNN_TENSORMEMTYPE_RAW; t->v2.clientBuf.data=s->buffers[slot]; t->v2.clientBuf.dataSize=size;
  }
  return 0;
}
void *q6a_create(const char *context_path, const char *library_dir, char *error, size_t error_size) {
  Session *s=calloc(1,sizeof(*s));
  if (!s) {snprintf(error,error_size,"session allocation failed");return NULL;}
  const QnnInterface_t **providers=NULL; uint32_t count=0; Qnn_ErrorHandle_t status;
  s->stub_lib=open_library(library_dir,"libQnnHtpV68Stub.so");
  s->htp_lib=open_library(library_dir,"libQnnHtp.so");
  s->system_lib=open_library(library_dir,"libQnnSystem.so");
  if (!s->stub_lib || !s->htp_lib || !s->system_lib) {
    const char *why=dlerror(); snprintf(s->error,sizeof(s->error),"QNN dlopen: %s",why?why:"missing library");goto failed;
  }
  QnnInterface_GetProvidersFn_t get=(QnnInterface_GetProvidersFn_t)dlsym(s->htp_lib,"QnnInterface_getProviders");
  if (!get || get(&providers,&count)) {fail(s,"HTP providers",0);goto failed;}
  unsigned selected=count;
  for (unsigned i=0;i<count;i++) if (providers[i]->backendId==QNN_BACKEND_ID_HTP &&
      providers[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR &&
      providers[i]->apiVersion.coreApiVersion.minor>=QNN_API_VERSION_MINOR) {selected=i;break;}
  if (selected==count) {fail(s,"compatible HTP provider required",count);goto failed;}
  s->api=providers[selected]->QNN_INTERFACE_VER_NAME;
  QnnSystemInterface_GetProvidersFn_t get_sys=(QnnSystemInterface_GetProvidersFn_t)dlsym(s->system_lib,"QnnSystemInterface_getProviders");
  const QnnSystemInterface_t **sp=NULL;
  if (!get_sys || get_sys(&sp,&count) || !count) {fail(s,"System providers",0);goto failed;}
  selected=count;
  for (unsigned i=0;i<count;i++) if (sp[i]->systemApiVersion.major==QNN_SYSTEM_API_VERSION_MAJOR &&
      sp[i]->systemApiVersion.minor>=QNN_SYSTEM_API_VERSION_MINOR) {selected=i;break;}
  if (selected==count) {fail(s,"compatible System provider required",count);goto failed;}
  s->sys=sp[selected]->QNN_SYSTEM_INTERFACE_VER_NAME;
#define CHECK(step, call) do {status=(call);if(status){fail(s,step,status);goto failed;}}while(0)
  CHECK("logCreate",s->api.logCreate(log_callback,QNN_LOG_LEVEL_WARN,&s->logger));
  CHECK("backendCreate",s->api.backendCreate(s->logger,NULL,&s->backend));
  CHECK("deviceCreate",s->api.deviceCreate(s->logger,NULL,&s->device));
  FILE *file=fopen(context_path,"rb");
  if (!file) {fail(s,"context open",0);goto failed;}
  if (fseek(file,0,SEEK_END)) {fclose(file);fail(s,"context seek",0);goto failed;}
  long length=ftell(file);
  if (length<=0 || length>128*1024*1024 || fseek(file,0,SEEK_SET)) {fclose(file);fail(s,"context size",0);goto failed;}
  s->binary=malloc((size_t)length);
  if (!s->binary || fread(s->binary,1,(size_t)length,file)!=(size_t)length) {fclose(file);fail(s,"context read",0);goto failed;}
  fclose(file);
  CHECK("systemContextCreate",s->sys.systemContextCreate(&s->metadata));
  const QnnSystemContext_BinaryInfo_t *info=NULL; Qnn_ContextBinarySize_t info_size=0;
  CHECK("systemContextGetBinaryInfo",s->sys.systemContextGetBinaryInfo(s->metadata,s->binary,length,&info,&info_size));
  QnnSystemContext_GraphInfo_t *graphs=NULL; uint32_t n_graphs=0,backend_id=0;
  if (info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graphs=info->contextBinaryInfoV1.graphs;n_graphs=info->contextBinaryInfoV1.numGraphs;backend_id=info->contextBinaryInfoV1.backendId;
  } else if (info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graphs=info->contextBinaryInfoV2.graphs;n_graphs=info->contextBinaryInfoV2.numGraphs;backend_id=info->contextBinaryInfoV2.backendId;
  } else if (info->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graphs=info->contextBinaryInfoV3.graphs;n_graphs=info->contextBinaryInfoV3.numGraphs;backend_id=info->contextBinaryInfoV3.backendId;
  }
  if (!graphs || n_graphs!=1 || backend_id!=QNN_BACKEND_ID_HTP) {fail(s,"one HTP graph required",n_graphs);goto failed;}
  const char *name=NULL;Qnn_Tensor_t *inputs=NULL,*outputs=NULL;uint32_t ni=0,no=0;
#define GRAPH_FIELDS(v) name=graphs[0].v.graphName;inputs=graphs[0].v.graphInputs;outputs=graphs[0].v.graphOutputs;ni=graphs[0].v.numGraphInputs;no=graphs[0].v.numGraphOutputs
  switch(graphs[0].version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: GRAPH_FIELDS(graphInfoV1);break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: GRAPH_FIELDS(graphInfoV2);break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: GRAPH_FIELDS(graphInfoV3);break;
    default:fail(s,"graph metadata version",graphs[0].version);goto failed;
  }
  if (!name || strcmp(name,"puzzle_int8_split_pc") || ni!=1 || no!=4) {fail(s,"unexpected graph",no);goto failed;}
  s->input=inputs[0];memcpy(s->outputs,outputs,sizeof(s->outputs));
  const uint32_t idims[]={1,3,640,640};
  if (bind_tensor(s,&s->input,0,"images",idims,4)) goto failed;
  const char *names[]={"boxes","scores","mask_coefficients","prototypes"};
  const uint32_t odims[][4]={{1,4,8400,0},{1,1,8400,0},{1,32,8400,0},{1,32,160,160}};
  for (unsigned k=0;k<4;k++) {
    unsigned j=0;for(;j<4;j++) if (FIELD(s->outputs[j],name) && !strcmp(FIELD(s->outputs[j],name),names[k])) break;
    if (j==4) {fail(s,"missing output",k);goto failed;}
    s->output_index[k]=j;
    if (bind_tensor(s,&s->outputs[j],j+1,names[k],odims[k],k==3?4:3)) goto failed;
  }
  CHECK("contextCreateFromBinary",s->api.contextCreateFromBinary(s->backend,s->device,NULL,s->binary,length,&s->context,NULL));
  CHECK("graphRetrieve",s->api.graphRetrieve(s->context,name,&s->graph));
  return s;
failed:
  snprintf(error,error_size,"%s",s->error);q6a_destroy(s);return NULL;
}
const char *q6a_error(void *opaque) {return opaque?((Session*)opaque)->error:"null session";}
int q6a_execute(void *opaque,const uint8_t *input,size_t input_size,float *boxes,float *scores,
                float *coefficients,float *prototypes,double *network_ms) {
  Session *s=opaque;
  if (!s || !input || input_size!=3*640*640 || !boxes || !scores || !coefficients || !prototypes || !network_ms) return -1;
  memcpy(s->buffers[0],input,input_size);
  struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);
  Qnn_ErrorHandle_t status=s->api.graphExecute(s->graph,&s->input,1,s->outputs,4,NULL,NULL);
  clock_gettime(CLOCK_MONOTONIC,&b);
  *network_ms=(b.tv_sec-a.tv_sec)*1000.0+(b.tv_nsec-a.tv_nsec)/1e6;
  if(status) return fail(s,"HTP graphExecute",status);
  float *destinations[]={boxes,scores,coefficients,prototypes};
  for(unsigned k=0;k<4;k++) {
    unsigned j=s->output_index[k];Qnn_Tensor_t t=s->outputs[j];
    Qnn_ScaleOffset_t quant=FIELD(t,quantizeParams).scaleOffsetEncoding;
    unsigned n=FIELD(t,clientBuf).dataSize;
    for(unsigned i=0;i<n;i++) destinations[k][i]=((int)s->buffers[j+1][i]+quant.offset)*quant.scale;
  }
  return 0;
}
