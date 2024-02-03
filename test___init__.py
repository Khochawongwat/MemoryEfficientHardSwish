import unittest
import torch
import torch.nn.functional as F

from __init__ import HardSwish

class TestHardSwish(unittest.TestCase):
    def setUp(self):
        self.hardswish = HardSwish()
        self.input_data = torch.rand((1, 3, 224, 224))

    def test_forward(self):
        # Test that the forward pass produces the expected output
        expected_output = F.hardswish(self.input_data.clone())
        actual_output = self.hardswish(self.input_data.clone())
        self.assertTrue(torch.allclose(expected_output, actual_output))

    def test_backward(self):
        # Test that the backward pass produces the expected gradients
        input_data = self.input_data.clone().requires_grad_()
        expected_output = F.hardswish(input_data)
        expected_output.sum().backward()
        expected_grad = input_data.grad.clone()
        input_data.grad.zero_()

        input_data = self.input_data.clone().requires_grad_()
        actual_output = self.hardswish(input_data)
        actual_output.sum().backward()
        actual_grad = input_data.grad.clone()

        self.assertTrue(torch.allclose(expected_grad, actual_grad))

    def test_dtype(self):
        # Test that the function works with different data types
        for dtype in [torch.float64, torch.float32, torch.float16]:
            input_data = self.input_data.to(dtype)
            output = self.hardswish(input_data)
            self.assertEqual(output.dtype, dtype)

if __name__ == '__main__':
    unittest.main()