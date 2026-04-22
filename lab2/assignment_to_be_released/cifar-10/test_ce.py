from train import MyCELoss
import torch
import torch.nn as nn


# 测试用例 1: 简单情况
print("=" * 50)
print("Test 1: Simple case with 2 classes")
pred = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
gt = torch.tensor([1, 0])

my_loss = MyCELoss(pred, gt)
pt_loss = nn.CrossEntropyLoss()(pred, gt)

print(f"My CE Loss: {my_loss.item():.6f}")
print(f"PyTorch CE Loss: {pt_loss.item():.6f}")
print(f"Difference: {abs(my_loss.item() - pt_loss.item()):.8f}")
print(f"Pass" if abs(my_loss.item() - pt_loss.item()) < 1e-5 else "Fail")

# 测试用例 2: 随机数据
print("\n" + "=" * 50)
print("Test 2: Random data")
torch.manual_seed(42)
pred = torch.randn(4, 10)
gt = torch.randint(0, 10, (4,))

my_loss = MyCELoss(pred, gt)
pt_loss = nn.CrossEntropyLoss()(pred, gt)

print(f"My CE Loss: {my_loss.item():.6f}")
print(f"PyTorch CE Loss: {pt_loss.item():.6f}")
print(f"Difference: {abs(my_loss.item() - pt_loss.item()):.8f}")
print(f"Pass" if abs(my_loss.item() - pt_loss.item()) < 1e-5 else "Fail")

# 测试用例 3: 边界情况
print("\n" + "=" * 50)
print("Test 3: Extreme values")
pred = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
gt = torch.tensor([0, 1])

my_loss = MyCELoss(pred, gt)
pt_loss = nn.CrossEntropyLoss()(pred, gt)

print(f"My CE Loss: {my_loss.item():.6f}")
print(f"PyTorch CE Loss: {pt_loss.item():.6f}")
print(f"Difference: {abs(my_loss.item() - pt_loss.item()):.8f}")
print(f"Pass" if abs(my_loss.item() - pt_loss.item()) < 1e-5 else "Fail")

# 测试用例 4: 数值稳定性
print("\n" + "=" * 50)
print("Test 4: Numerical stability")
pred = torch.tensor([[1000.0, 1000.0, 1000.0]])
gt = torch.tensor([1])

my_loss = MyCELoss(pred, gt)
pt_loss = nn.CrossEntropyLoss()(pred, gt)

print(f"My CE Loss: {my_loss.item():.6f}")
print(f"PyTorch CE Loss: {pt_loss.item():.6f}")
print(f"Difference: {abs(my_loss.item() - pt_loss.item()):.8f}")
print(f"Pass" if abs(my_loss.item() - pt_loss.item()) < 1e-5 else "Fail")
