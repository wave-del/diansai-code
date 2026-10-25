"""QNN decoding and resource failures; these unit tests never initialize hardware."""
import ctypes
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vision/src'))
from ec_26_e.vision.contracts import VisionConfig
from ec_26_e.vision.qnn import OUTPUT_SHAPES, QnnSession, decode, letterbox


class DecoderTests(unittest.TestCase):
    def outputs(self): return tuple(np.zeros(s,np.float32) for s in OUTPUT_SHAPES)

    def test_letterbox_rgb_padding_layout_and_odd_rounding(self):
        image=np.zeros((301,640,3),np.uint8);image[:]=(11,22,33)
        data=letterbox(image)
        self.assertEqual(data.shape,(1,3,640,640))
        self.assertTrue(data.flags.c_contiguous)
        np.testing.assert_array_equal(data[0,:,168,0],[114,114,114])
        np.testing.assert_array_equal(data[0,:,169,0],[33,22,11])
        np.testing.assert_array_equal(data[0,:,469,0],[33,22,11])
        np.testing.assert_array_equal(data[0,:,470,0],[114,114,114])

    def test_no_detections_and_malformed_outputs(self):
        config=VisionConfig()
        self.assertEqual(decode(self.outputs(),(540,764),config),())
        with self.assertRaises(ValueError):decode(self.outputs()[:2],(540,764),config)
        values=self.outputs();values[1][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'非有限'):decode(values,(540,764),config)

    def test_nms_before_masks_preserves_more_than_four_separate_instances(self):
        values=self.outputs();boxes,scores,coeff,proto=values
        proto[0,0]=1
        for i in range(5):
            boxes[0,:,i]=[60+i*115,320,50,80];scores[0,0,i]=.9-i*.01;coeff[0,0,i]=1
        # Sixth proposal is a duplicate of the first, despite a distinct anchor.
        boxes[0,:,5]=boxes[0,:,0];scores[0,0,5]=.8;coeff[0,0,5]=1
        instances=decode(values,(640,640),VisionConfig())
        self.assertEqual(len(instances),5)
        self.assertEqual(instances[0].class_id,0)
        self.assertEqual(int(instances[0].mask.sum()),50*80)
        self.assertEqual(instances[0].mask.dtype,np.uint8)

    def test_box_and_mask_inverse_letterbox_coordinates(self):
        boxes,scores,coeff,proto=self.outputs()
        boxes[0,:,0]=[320,320,100,100];scores[0,0,0]=.8;coeff[0,0,0]=1;proto[0,0]=1
        result=decode((boxes,scores,coeff,proto),(320,640),VisionConfig())[0]
        np.testing.assert_allclose(result.box_px,(270,110,370,210))
        self.assertEqual(int(result.mask.sum()),10000)
        self.assertFalse(result.mask[:110].any())


class BundleTests(unittest.TestCase):
    def test_incomplete_manifest_fails_before_loading_any_native_library(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctypes,'CDLL',side_effect=AssertionError('loaded')):
            p=Path(tmp);(p/'manifest.json').write_text(json.dumps({'backend':'QNN HTP','cpu_fallback':False,'sha256':{}}))
            with self.assertRaisesRegex(RuntimeError,'缺少'):QnnSession(p)

    def test_missing_or_cpu_manifest_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctypes,'CDLL',side_effect=AssertionError('loaded')):
            p=Path(tmp)
            with self.assertRaises(FileNotFoundError):QnnSession(p)
            (p/'manifest.json').write_text('{"backend":"CPU","cpu_fallback":true}')
            with self.assertRaisesRegex(RuntimeError,'HTP'):QnnSession(p)


if __name__=='__main__':unittest.main()
