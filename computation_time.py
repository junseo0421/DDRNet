import torch
import time
from DDRNet import DDRNet
from thop import profile
import statistics as stats

modelName = "DDRNet"
numClasses = 19
compute_speed = True

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')
model = eval(modelName)(num_classes=numClasses).eval().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1024 **2:.3f} M")

# dummy input
x = torch.randn(1, 3, 1080, 1920, device=device)

# FLOPs
flops, params = profile(model, inputs=(x,))
print(f'Parameters: {params/(1024*1024):.2f} M')
result = f'FLOPs: {flops/(1024*1024*1024):.2f} G'
print(result)

with open("FLOPs.txt", "w") as f:
    f.write(result + "\n")

# memory usage (rough)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()
mem_used = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak GPU Memory: {mem_used:.2f} MB")

if compute_speed:
    # warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()

    results = []
    iterations = 200  

    for _ in range(9):  
        torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        torch.cuda.synchronize()
        t_end = time.time()

        elapsed = t_end - t_start
        latency_ms = (elapsed / iterations) * 1000
        results.append(latency_ms)

    mean_ms = sum(results) / len(results)
    median_ms = stats.median(results)
    fps_mean = 1000.0 / mean_ms
    fps_median = 1000.0 / median_ms

    print("\n========= Speed (per forward) =========")
    print(f"Per-run latencies (ms): {[round(v,3) for v in results]}")
    print(f"Median: {median_ms:.3f} ms   (~{fps_median:.2f} FPS)")
