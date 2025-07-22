# PDF2Image Optimizer

🚀 **High-performance PDF to image converter optimized for Windows threading issues**

A drop-in replacement and optimization layer for `pdf2image` that solves Windows threading bottlenecks and provides multiple conversion strategies for different use cases.
I think the real issue would be to investigate in poppler code itself, however, this is a  quick  solution to memory error or slow conversion

## 🎯 Problem Solved

The popular `pdf2image` library suffers from significant performance issues on Windows when using multiple threads (often 2-4x slower than single-threading). This optimizer:

- ✅ **Fixes Windows threading bottlenecks** - Automatically detects and optimizes for Windows
- ✅ **faster conversions** on Windows systems
- ✅ **Multiple optimization strategies** for different file sizes and use cases
- ✅ **Drop-in replacement** - Works with existing `pdf2image` code
- ✅ **Memory-efficient** processing for large PDFs
- ✅ **Auto-detection** of optimal conversion strategy

## 📊 Performance Comparison

📊 Speed improvement vs slowest: 2.7x  

results:  
{'single': {'time': 878.5427498817444, 'pages': 3096, 'success': True, 'pages_per_second': 3.5240174714511445}, 
'batch': {'time': 1024.443242073059, 'pages': 3091, 'success': True, 'pages_per_second': 3.017248660594476}, 
'async': {'time': 448.80277943611145, 'pages': 3096, 'success': True, 'pages_per_second': 6.898353000152767}, 
'memory': {'time': 1207.1273488998413, 'pages': 3096, 'success': True, 'pages_per_second': 2.564766677535432}}

*Benchmarked on Windows 10, 13th Gen Intel(R) Core(TM) i5-13500H, 32GB RAM*

## 🔧 Installation

```bash
pip install pdf2image pillow
```

**Additional Requirements:**
- **Windows**: Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH
- **Linux**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`

## 🚀 Quick Start
NB: I would suggest to launch benchmarks and then choose solution that will give best result suiting your documents

### Option 1: Drop-in Replacement (Recommended)
```python
from pdf_optimizer import apply_patch

# Apply optimization patch once
apply_patch()

# Now use pdf2image normally - it's automatically optimized!
from pdf2image import convert_from_path
images = convert_from_path("document.pdf")  # faster on Windows!
```

### Option 2: Direct Usage

```python
from pdf_optimizer import convert_pdf_fast

# Auto-optimized conversion
images = convert_pdf_fast("document.pdf", dpi=200, fmt="png")
print(f"Converted {len(images)} pages")
```

### Option 3: Memory-Efficient (Large Files - theoretically)

```python
from pdf_optimizer import convert_pdf_memory_efficient

# For large PDFs - saves to disk instead of loading into memory
paths = convert_pdf_memory_efficient("large_document.pdf", dpi=150)
```

## 🎛️ Advanced Usage

### Manual Strategy Selection

```python
from pdf_optimizer import OptimizedPDFConverter

converter = OptimizedPDFConverter()

# Choose specific optimization strategy
images = converter.convert_from_path_optimized(
    "document.pdf",
    optimization_strategy="batch",  # single, batch, async, memory
    batch_size=5,
    dpi=300,
    fmt="png"
)
```

### Benchmark Different Strategies

```python
from pdf_optimizer import benchmark_strategies

results = benchmark_strategies("test.pdf")

# Find fastest strategy
fastest = min((k for k, v in results.items() if v['success']), 
              key=lambda x: results[x]['time'])
print(f"Fastest strategy: {fastest}")
```

### Get PDF Information

```python
from pdf_optimizer import get_pdf_info

info = get_pdf_info("document.pdf")
print(f"Pages: {info['pages']}, Size: {info['size_mb']:.1f}MB")
```

## 🎯 Optimization Strategies

The optimizer automatically selects the best strategy, but you can choose manually:

| Strategy | Best For | Description |
|----------|----------|-------------|
| **`auto`** | All files | Auto-detects optimal strategy (recommended) |
| **`single`** | Small files (≤5 pages) | Single-threaded processing |
| **`batch`** | Medium files (6-20 pages) | Process in small batches |
| **`async`** | Unix systems - or windows ;-) | Async I/O optimization |
| **`memory`** | Large files (>50MB) | Memory-efficient streaming |

## 🪟 Windows-Specific Optimizations

The optimizer automatically applies these Windows optimizations:

- **Single Threading**: Forces `thread_count=1` to avoid subprocess overhead
- **pdftocairo**: Uses `pdftocairo` instead of `pdftoppm` (often faster on Windows)
- **Process Creation Flags**: Optimized subprocess creation with `CREATE_NO_WINDOW`
- **Batch Processing**: Automatic batching for large files to reduce memory pressure
- **Extended Timeouts**: Accounts for Windows process creation delays

## 📋 API Reference

### `convert_pdf_fast(pdf_path, **kwargs)`
Quick conversion with auto-optimization.
- **pdf_path**: Path to PDF file
- **kwargs**: Standard pdf2image arguments (dpi, fmt, output_folder, etc.)
- **Returns**: List of PIL Images

### `convert_pdf_memory_efficient(pdf_path, **kwargs)`
Memory-efficient conversion for large files.
- **Returns**: List of file paths (images saved to disk)

### `apply_patch()`
Applies optimization patch to the original `pdf2image.convert_from_path`.
- **Returns**: Boolean indicating success

### `benchmark_strategies(pdf_path, strategies=None)`
Benchmark different conversion strategies.
- **strategies**: List of strategies to test (default: all)
- **Returns**: Dictionary with performance results

## 🔍 Troubleshooting

### Common Issues

**1. `ModuleNotFoundError: No module named 'pdf2image'`**
```bash
pip install pdf2image pillow
```

**2. `Unable to get page count from pdf path`**
Install a PDF library for fallback page detection:
```bash
pip install PyPDF2
# or
pip install pypdf
# or  
pip install pdfplumber
```

**3. `pdftoppm not found`**
Install Poppler:
- **Windows**: Download from [Poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)
- **Linux**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`

**4. Still slow on Windows?**
Make sure the patch is applied:
```python
from pdf_optimizer import apply_patch
success = apply_patch()
print(f"Patch applied: {success}")
```

## 🧪 Testing

Run the built-in tests:

```python
python pdf_optimizer.py
```

This will:
- Check all dependencies
- Display optimization strategies
- Run benchmarks if a test PDF is found


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf2image-optimizer.git
cd pdf2image-optimizer
```

2. Install dependencies:
```bash
pip install pdf2image pillow PyPDF2
```

3. Run tests:
```bash
python pdf_optimizer.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

## 🔗 Related Projects

- [pdf2image](https://github.com/Belval/pdf2image) - The original library this optimizes
- [Poppler](https://poppler.freedesktop.org/) - The underlying PDF rendering library
- [Pillow](https://github.com/python-pillow/Pillow) - Python Imaging Library

## 📞 Support

- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and community support
- **Email**: For private inquiries (add your contact info)

---

**Made with ❤️ for the Python community**

*Tired of slow PDF conversions on Windows? This optimizer has got you covered!* 🚀
