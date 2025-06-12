import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam

# def get_soft_cosine_schedule_with_smooth_decay(
#     optimizer, 
#     num_warmup_steps, 
#     num_training_steps, 
#     num_cycles=4,
#     decay_factor=0.1,     # 最终峰值下降到初始的 10%
#     last_epoch=-1
# ):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
        
#         progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
#         # 平滑指数衰减，progress 从 0 → 1，decay 从 1 → decay_factor
#         decay_amplitude = decay_factor ** progress
        
#         # 改成 π，而非 2π，使得最后 step = 1 时余弦值为 -1 → 最终 lr 为 0
#         cosine = 0.5 * (1 + math.cos(math.pi * num_cycles * progress))
        
#         return max(0.0, decay_amplitude * cosine)
    
#     return LambdaLR(optimizer, lr_lambda, last_epoch)
def get_soft_cosine_schedule_with_smooth_decay(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=4,
    decay_factor=0.1,  # 最后衰减到的相对幅度
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # 多周期余弦震荡
        cosine_osc = 0.5 * (1 + math.cos(2 * math.pi * num_cycles * progress))
        
        # 平滑包络衰减函数（指数衰减）
        envelope_decay = decay_factor ** progress
        
        # 尾部强制衰减为 0（额外乘一个余弦窗，从 1 到 0）
        end_window = 0.5 * (1 + math.cos(math.pi * progress))  # progress=1 时为0
        
        return max(0.0, envelope_decay * cosine_osc * end_window)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)




# ===== 参数设置 =====
num_warmup_steps = 500      # Warmup 步数
num_training_steps = 10000  # 总训练步数
num_cycles = 3              # 4 个余弦周期
decay_factor = 0.3        # 每个周期峰值衰减 10%
initial_lr = 0.05          # 初始学习率

# ===== 模拟优化器和调度器 =====
optimizer = Adam([{'params': []}], lr=initial_lr)  # 虚拟优化器
scheduler = get_soft_cosine_schedule_with_smooth_decay(
    optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    decay_factor=decay_factor
)

# ===== 记录学习率变化 =====
learning_rates = []
for step in range(num_training_steps):
    optimizer.step()  # 模拟优化步骤
    scheduler.step()  # 更新学习率
    learning_rates.append(optimizer.param_groups[0]['lr'])

# ===== 绘制学习率曲线 =====
plt.figure(figsize=(12, 6))
plt.plot(range(num_training_steps), learning_rates, linewidth=2, color='royalblue')

# 标记 Warmup 结束
plt.axvline(x=num_warmup_steps, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
plt.text(
    num_warmup_steps + 100, initial_lr * 0.2, 
    f'Warmup Ends\n({num_warmup_steps} steps)', 
    color='red', fontsize=10
)

# 标记余弦周期峰值
for cycle in range(1, num_cycles + 1):
    cycle_length = (num_training_steps - num_warmup_steps) / num_cycles
    peak_step = num_warmup_steps + (cycle - 0.5) * cycle_length
    peak_lr = initial_lr * (decay_factor **(cycle - 1))  # 衰减后的峰值
    plt.scatter(peak_step, peak_lr, color='orange', s=80, zorder=5)
    plt.text(
        peak_step + 100, peak_lr * 1.05, 
        f'Cycle {cycle}\nLR={peak_lr:.2e}', 
        fontsize=9, color='darkorange'
    )

# 图表美化
plt.title(
    f"Soft Cosine LR Schedule (Decay Factor={decay_factor}, {num_cycles} Cycles)",
    fontsize=14, pad=20
)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.grid(True, alpha=0.4)
plt.tight_layout()

plt.show()