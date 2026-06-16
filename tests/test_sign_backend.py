import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
from core.controller import CentralController, AppMode
from sign_language.backend import SignDetectionBackend

def test_sign_backend_init():
    backend = SignDetectionBackend()
    assert backend.module_name == "SignDetectionBackend"
    assert not backend.is_running
    assert backend.is_healthy
    assert backend.on_sentence_callback is None
    assert backend.is_active_callback is None

@patch('sign_language.backend.torch.load')
@patch('sign_language.backend.vision.HandLandmarker.create_from_options')
def test_sign_backend_start_stop(mock_landmarker, mock_torch_load):
    backend = SignDetectionBackend()
    
    mock_model = MagicMock()
    with patch('sign_language.backend.AlphabetTCN', return_value=mock_model):
        res = backend.start()
        assert res is True
        assert backend.is_running
        assert backend.is_healthy
        
        backend.stop()
        assert not backend.is_running

@patch('sign_language.backend.torch.load')
@patch('sign_language.backend.vision.HandLandmarker.create_from_options')
def test_sign_backend_on_frame_and_gating(mock_landmarker, mock_torch_load):
    backend = SignDetectionBackend()
    
    mock_model = MagicMock()
    with patch('sign_language.backend.AlphabetTCN', return_value=mock_model):
        backend.start()
        
        # Setup is_active_callback to return False
        backend.is_active_callback = MagicMock(return_value=False)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        backend.on_frame(frame)
        
        # The queue should be empty because it is not active
        assert backend._frame_queue.empty()
        
        # Setup is_active_callback to return True
        backend.is_active_callback = MagicMock(return_value=True)
        backend.on_frame(frame)
        
        # The queue should now have 1 item
        assert backend._frame_queue.qsize() == 1
        
        backend.stop()

def test_sign_module_registration():
    controller = CentralController()
    mock_module = MagicMock()
    controller.register_sign_module(mock_module)
    assert controller._sign_module == mock_module
    assert mock_module.on_sentence_callback == controller.send_sign_translation
    
    # Verify is_active_callback
    assert mock_module.is_active_callback() is False  # AppMode.DANGER by default
    controller.set_app_mode("sign")
    assert mock_module.is_active_callback() is True

def test_controller_send_sign_translation():
    controller = CentralController()
    translations_sent = []
    class MockFlutterServer:
        def send_sign_translation(self, sentence):
            translations_sent.append(sentence)
    controller.set_flutter_server(MockFlutterServer())
    controller.send_sign_translation("HELLO WORLD")
    assert translations_sent == ["HELLO WORLD"]

def test_controller_start_stop_sign_module():
    controller = CentralController()
    mock_module = MagicMock()
    controller.register_sign_module(mock_module)
    
    controller._start_guest_modules()
    mock_module.start.assert_called_once()
    
    controller._stop_guest_modules()
    mock_module.stop.assert_called_once()

def test_controller_distribute_frame_to_sign_module():
    controller = CentralController()
    mock_module = MagicMock()
    controller.register_sign_module(mock_module)
    
    frame = MagicMock()
    controller._distribute_frame(frame)
    mock_module.on_frame.assert_called_once_with(frame)
