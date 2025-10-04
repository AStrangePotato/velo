import unittest
import numpy as np
import cv2

# Mock functions and classes for testing without full dependencies
# In a real scenario, you might use a more sophisticated mocking library

# Assuming preprocessing.py is in the same directory or in PYTHONPATH
from preprocessing import (
    convert_to_grayscale,
    resize_frame,
    denoise_frame,
    subtract_background,
    create_background_subtractor,
    get_video_properties
)

class MockVideoCapture:
    def __init__(self, is_opened=True, width=1920, height=1080, fps=30.0):
        self._is_opened = is_opened
        self._properties = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps
        }

    def isOpened(self):
        return self._is_opened

    def get(self, prop_id):
        return self._properties.get(prop_id, 0)

    def read(self):
        if not self._is_opened:
            return False, None
        # Return a consistent dummy frame for testing
        frame = np.zeros((self._properties[cv2.CAP_PROP_FRAME_HEIGHT], self._properties[cv2.CAP_PROP_FRAME_WIDTH], 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self._is_opened = False


class TestPreprocessing(unittest.TestCase):
    """
    Test suite for the preprocessing module.
    This class contains a series of tests to ensure that the image and video
    preprocessing functions work as expected. It uses mock data to isolate
    the functions from external dependencies like actual video files.
    """

    def setUp(self):
        """
        Set up a consistent test environment for each test case.
        This method is called before each test function is executed.
        It creates a standard mock color image frame.
        """
        self.height = 720
        self.width = 1280
        # Create a dummy 3-channel color image (H, W, C)
        self.mock_frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        # Create a single channel grayscale image
        self.mock_gray_frame = np.random.randint(0, 256, (self.height, self.width), dtype=np.uint8)

    def test_get_video_properties(self):
        """
        Test that video properties (width, height, fps) are read correctly.
        """
        mock_cap = MockVideoCapture(width=1920, height=1080, fps=29.97)
        width, height, fps = get_video_properties(mock_cap)
        self.assertEqual(width, 1920)
        self.assertEqual(height, 1080)
        self.assertAlmostEqual(fps, 29.97, places=2)

    def test_convert_to_grayscale(self):
        """
        Test the conversion of a BGR frame to grayscale.
        Ensures the output is a single-channel image with the correct dimensions.
        """
        gray_frame = convert_to_grayscale(self.mock_frame)
        self.assertIsNotNone(gray_frame)
        self.assertEqual(len(gray_frame.shape), 2)  # Should be 2D (H, W)
        self.assertEqual(gray_frame.shape[0], self.height)
        self.assertEqual(gray_frame.shape[1], self.width)
        self.assertEqual(gray_frame.dtype, np.uint8)

    def test_resize_frame(self):
        """
        Test the frame resizing functionality.
        Checks if the output frame has the specified dimensions.
        """
        target_width = 640
        target_height = 360
        resized = resize_frame(self.mock_frame, target_width, target_height)
        self.assertIsNotNone(resized)
        self.assertEqual(resized.shape[0], target_height)
        self.assertEqual(resized.shape[1], target_width)
        self.assertEqual(resized.shape[2], 3) # Should remain a 3-channel image

    def test_resize_grayscale_frame(self):
        """
        Test resizing on a grayscale image.
        """
        target_width = 320
        target_height = 180
        resized_gray = resize_frame(self.mock_gray_frame, target_width, target_height)
        self.assertIsNotNone(resized_gray)
        self.assertEqual(resized_gray.shape[0], target_height)
        self.assertEqual(resized_gray.shape[1], target_width)
        self.assertEqual(len(resized_gray.shape), 2)

    def test_denoise_frame(self):
        """
        Test the denoising (Gaussian blur) functionality.
        The output should have the same dimensions and type as the input.
        """
        denoised_frame = denoise_frame(self.mock_frame)
        self.assertIsNotNone(denoised_frame)
        self.assertEqual(denoised_frame.shape, self.mock_frame.shape)
        self.assertEqual(denoised_frame.dtype, self.mock_frame.dtype)
        # Check that the image is not the same (blur should change pixel values)
        self.assertFalse(np.array_equal(denoised_frame, self.mock_frame))

    def test_create_background_subtractor(self):
        """
        Test the creation of a background subtractor object.
        Ensures it returns a valid OpenCV BackgroundSubtractorMOG2 object.
        """
        subtractor = create_background_subtractor()
        self.assertIsNotNone(subtractor)
        # Check if it's an instance of the expected OpenCV class
        self.assertIsInstance(subtractor, type(cv2.createBackgroundSubtractorMOG2()))

    def test_subtract_background(self):
        """
        Test the background subtraction process.
        The output should be a single-channel foreground mask.
        """
        subtractor = create_background_subtractor()
        # Apply a few frames to initialize the background model
        subtractor.apply(self.mock_frame)
        subtractor.apply(self.mock_frame)
        foreground_mask = subtract_background(self.mock_frame, subtractor)
        self.assertIsNotNone(foreground_mask)
        self.assertEqual(len(foreground_mask.shape), 2) # Mask is 2D
        self.assertEqual(foreground_mask.shape, (self.height, self.width))
        self.assertEqual(foreground_mask.dtype, np.uint8)

    def test_subtract_background_no_subtractor(self):
        """
        Test that background subtraction raises an error if the subtractor is None.
        """
        with self.assertRaises(ValueError):
            subtract_background(self.mock_frame, None)

    def tearDown(self):
        """
        Clean up after each test.
        This method is called after each test function is executed.
        Here, it just nullifies the mock data.
        """
        self.mock_frame = None
        self.mock_gray_frame = None

if __name__ == '__main__':
    """
    Standard entry point to run the tests.
    This allows the script to be executed directly to run the test suite.
    """
    unittest.main()
