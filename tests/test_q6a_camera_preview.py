"""Preview HTTP/read-only contract without opening a camera or the NPU."""
import http.client
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'npu_q6a_test/web_preview/app'))
from q6a_camera_preview import Preview,Server,handler_for


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.preview=Preview(SimpleNamespace(camera='/dev/test-only'))
        self.server=Server(('127.0.0.1',0),handler_for(self.preview))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def tearDown(self):
        self.preview.stop.set()
        with self.preview.changed:self.preview.changed.notify_all()
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)

    def connection(self):return http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)

    def test_constructing_preview_does_not_initialize_devices(self):
        with patch('q6a_camera_preview.cv2.VideoCapture') as camera,patch('q6a_camera_preview.QnnYolo11SegInferencer') as npu:
            preview=Preview(SimpleNamespace(camera='/dev/test-only'))
            self.assertFalse(preview.snapshot()['motion_enabled'])
            camera.assert_not_called();npu.assert_not_called()

    def test_status_and_page_expose_visual_only_state(self):
        connection=self.connection();connection.request('GET','/api/status');response=connection.getresponse()
        status=json.loads(response.read());self.assertEqual(response.status,200)
        self.assertFalse(status['motion_enabled']);self.assertTrue(status['solver_enabled'])
        self.assertFalse(status['cpu_fallback']);self.assertTrue(status['preview_only'])
        connection.request('GET','/');response=connection.getresponse();html=response.read().decode()
        self.assertIn('/stream.mjpg?view=overlay',html);self.assertIn('运动禁用',html);connection.close()

    def test_unpublished_snapshot_is_not_silently_a_success(self):
        connection=self.connection();connection.request('GET','/snapshot.jpg?view=raw');response=connection.getresponse()
        self.assertEqual(response.status,503);response.read()
        data=self.preview.images['rectified'];self.preview.publish('raw',data)
        connection.request('GET','/snapshot.jpg?view=raw');response=connection.getresponse()
        self.assertEqual(response.status,200);self.assertEqual(response.getheader('Content-Type'),'image/jpeg')
        self.assertEqual(response.read(),data);connection.close()

    def test_stream_delivers_new_frames_not_the_same_buffer_forever(self):
        first=self.preview.images['rectified'];second=first+b'new-frame-marker'
        self.preview.publish('raw',first)
        connection=self.connection();connection.request('GET','/stream.mjpg?view=raw');response=connection.getresponse()
        self.assertIn('multipart/x-mixed-replace',response.getheader('Content-Type'))
        def frame():
            self.assertEqual(response.readline(),b'--frame\r\n')
            self.assertEqual(response.readline(),b'Content-Type: image/jpeg\r\n')
            length=int(response.readline().split(b':')[1]);self.assertEqual(response.readline(),b'\r\n')
            result=response.read(length);self.assertEqual(response.read(2),b'\r\n');return result
        self.assertEqual(frame(),first);self.preview.publish('raw',second);self.assertEqual(frame(),second)
        connection.close()

    def test_motion_routes_and_arbitrary_paths_are_unavailable(self):
        for target in ['/api/enable-motion','/../../q6a_config.json','/snapshot.jpg?view=../../secret']:
            connection=self.connection();connection.request('GET',target);response=connection.getresponse()
            self.assertEqual(response.status,404);response.read();connection.close()
        connection=self.connection();connection.request('POST','/api/enable-motion',body=b'{}');response=connection.getresponse()
        self.assertEqual(response.status,404);response.read();connection.close()



if __name__=='__main__':unittest.main()
