#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete PDF to Image Converter Optimizer - Standalone Version
Works without accessing internal pdf2image utilities

Save this as: pdf_optimizer.py
Usage: 
    from pdf_optimizer import convert_pdf_fast, apply_patch
    
    # Quick usage
    images = convert_pdf_fast("file.pdf")
    
    # Or apply patch
    apply_patch()
    from pdf2image import convert_from_path
    images = convert_from_path("file.pdf")  # Now optimized!
"""

import os
import platform
import tempfile
import shutil
import subprocess
import threading
import queue
import time
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union, Tuple, Any, Dict, Optional
import multiprocessing as mp

# Try to import pdf2image
try:
    from pdf2image import convert_from_path as original_convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pdf2image not available: {e}")
    print("Install with: pip install pdf2image pillow")
    PDF2IMAGE_AVAILABLE = False
    original_convert_from_path = None

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not available. Install with: pip install pillow")
    PIL_AVAILABLE = False


class PDFInfoExtractor:
    """Extract PDF information without relying on pdf2image internals"""

    @staticmethod
    def get_page_count(pdf_path: str, poppler_path: str = None) -> int:
        """Get page count using pdfinfo command"""
        try:
            # Try using pdfinfo command
            cmd = ["pdfinfo", pdf_path]
            if poppler_path:
                cmd = [os.path.join(poppler_path, "pdfinfo")] + [pdf_path]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Pages:'):
                        return int(line.split(':')[1].strip())

            # Fallback: try with python-based PDF libraries
            return PDFInfoExtractor._get_page_count_fallback(pdf_path)

        except Exception:
            return PDFInfoExtractor._get_page_count_fallback(pdf_path)

    @staticmethod
    def _get_page_count_fallback(pdf_path: str) -> int:
        """Fallback methods to get page count"""
        # Try PyPDF2
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except ImportError:
            pass
        except Exception:
            pass

        # Try pypdf
        try:
            import pypdf
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                return len(reader.pages)
        except ImportError:
            pass
        except Exception:
            pass

        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except ImportError:
            pass
        except Exception:
            pass

        # Last resort: assume single page
        print(
            f"Warning: Could not determine page count for {pdf_path}, assuming 1 page")
        return 1


class OptimizedPDFConverter:
    """
    Optimized PDF to image converter with Windows-specific optimizations
    that doesn't rely on pdf2image internals
    """

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.cpu_count = os.cpu_count() or 4
        self.pdf2image_available = PDF2IMAGE_AVAILABLE

    def convert_from_path_optimized(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
        output_folder: Union[str, Path] = None,
        first_page: int = None,
        last_page: int = None,
        fmt: str = "ppm",
        optimization_strategy: str = "auto",
        batch_size: int = None,
        use_process_pool: bool = False,
        **kwargs
    ) -> List:
        """
        Optimized PDF conversion with multiple strategies

        Args:
            pdf_path: Path to PDF file
            dpi: Image resolution (default: 200)
            output_folder: Output directory (None = memory)
            first_page: First page to convert
            last_page: Last page to convert
            fmt: Output format (ppm, png, jpg, etc.)
            optimization_strategy: 
                - "auto": Auto-detect best strategy (recommended)
                - "single": Single-threaded (safest for Windows)
                - "batch": Process in batches (good for medium files)
                - "async": Async I/O optimization (better for Unix)
                - "memory": Memory-optimized streaming (large files)
            batch_size: Pages per batch (auto-calculated if None)
            use_process_pool: Use ProcessPoolExecutor instead of threading
            **kwargs: Additional arguments passed to pdf2image

        Returns:
            List of PIL Images or file paths (if paths_only=True)
        """

        if not self.pdf2image_available:
            raise ImportError(
                "pdf2image is required. Install with: pip install pdf2image pillow")

        pdf_path = str(pdf_path)  # Convert Path to string

        if optimization_strategy == "auto":
            optimization_strategy = self._detect_optimal_strategy(pdf_path)

        strategy_map = {
            "single": self._single_threaded_convert,
            "batch": self._batch_convert,
            "async": self._async_convert,
            "memory": self._memory_optimized_convert,
        }

        if optimization_strategy not in strategy_map:
            print(
                f"Unknown strategy '{optimization_strategy}', using 'single'")
            optimization_strategy = "single"

        return strategy_map[optimization_strategy](
            pdf_path, dpi, output_folder, first_page, last_page, fmt,
            batch_size, use_process_pool, **kwargs
        )

    def _detect_optimal_strategy(self, pdf_path: str) -> str:
        """Auto-detect the best optimization strategy based on file characteristics"""
        try:
            page_count = PDFInfoExtractor.get_page_count(pdf_path)

            try:
                file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
            except (OSError, FileNotFoundError):
                file_size = 10  # Default assumption

            print(f"📄 Detected: {page_count} pages, {file_size:.1f}MB")

            if self.is_windows:
                if page_count <= 5:
                    strategy = "single"
                elif page_count <= 20:
                    strategy = "batch"
                elif file_size > 50:  # Large files
                    strategy = "memory"
                else:
                    strategy = "batch"
            else:
                # Unix systems handle threading better
                if page_count <= 10:
                    strategy = "single"
                else:
                    strategy = "async"

            print(f"🎯 Selected strategy: {strategy}")
            return strategy

        except Exception as e:
            print(f"Strategy detection failed: {e}, using 'single'")
            return "single"  # Fallback to safest option

    def _single_threaded_convert(self, pdf_path, dpi, output_folder, first_page,
                                 last_page, fmt, batch_size, use_process_pool, **kwargs):
        """Optimized single-threaded conversion"""
        kwargs['thread_count'] = 1

        # Windows-specific optimizations
        if self.is_windows:
            # Often faster on Windows
            kwargs.setdefault('use_pdftocairo', True)
            print("🪟 Applied Windows optimizations: single thread, pdftocairo")

        return original_convert_from_path(
            pdf_path, dpi=dpi, output_folder=output_folder,
            first_page=first_page, last_page=last_page,
            fmt=fmt, **kwargs
        )

    def _batch_convert(self, pdf_path, dpi, output_folder, first_page,
                       last_page, fmt, batch_size, use_process_pool, **kwargs):
        """Process PDF in batches to reduce memory pressure"""

        page_count = PDFInfoExtractor.get_page_count(pdf_path)

        if first_page is None:
            first_page = 1
        if last_page is None:
            last_page = page_count

        # Auto-calculate batch size
        if batch_size is None:
            total_pages = last_page - first_page + 1
            if self.is_windows:
                batch_size = min(5, max(1, total_pages // 4))
            else:
                batch_size = min(10, max(1, total_pages // 2))

        all_images = []
        kwargs['thread_count'] = 1  # Single thread per batch

        if self.is_windows:
            kwargs.setdefault('use_pdftocairo', True)

        total_pages = last_page - first_page + 1
        print(f"📦 Processing {total_pages} pages in batches of {batch_size}")

        for start in range(first_page, last_page + 1, batch_size):
            end = min(start + batch_size - 1, last_page)

            print(f"   Processing pages {start}-{end}...")
            try:
                batch_images = original_convert_from_path(
                    pdf_path, dpi=dpi, output_folder=output_folder,
                    first_page=start, last_page=end, fmt=fmt, **kwargs
                )
                all_images.extend(batch_images)
                print(
                    f"   ✅ Completed pages {start}-{end} ({len(batch_images)} images)")

            except Exception as e:
                print(f"   ❌ Failed pages {start}-{end}: {e}")
                continue

            # Optional: Force garbage collection between batches for large files
            if len(all_images) > 50:
                import gc
                gc.collect()

        return all_images

    def _async_convert(self, pdf_path, dpi, output_folder, first_page,
                       last_page, fmt, batch_size, use_process_pool, **kwargs):
        """Async I/O optimized conversion using ThreadPoolExecutor"""

        page_count = PDFInfoExtractor.get_page_count(pdf_path)

        if first_page is None:
            first_page = 1
        if last_page is None:
            last_page = page_count

        # Calculate optimal batch size and worker count
        total_pages = last_page - first_page + 1
        if self.is_windows:
            # Conservative on Windows: use 25-50% of cores, max 4 workers
            max_workers = min(4, max(1, self.cpu_count // 2))
            batch_size = batch_size or max(1, total_pages // max_workers)
        else:
            max_workers = min(4, self.cpu_count)
            batch_size = batch_size or max(1, total_pages // max_workers)

        def convert_batch(start_page, end_page):
            batch_kwargs = kwargs.copy()
            batch_kwargs['thread_count'] = 1
            if self.is_windows:
                batch_kwargs.setdefault('use_pdftocairo', True)

            return original_convert_from_path(
                pdf_path, dpi=dpi, output_folder=output_folder,
                first_page=start_page, last_page=end_page,
                fmt=fmt, **batch_kwargs
            )

        all_images = []

        # Use ProcessPoolExecutor for CPU-bound work if requested
        executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

        print(
            f"⚡ Processing with {max_workers} workers, batch size {batch_size}")

        with executor_class(max_workers=max_workers) as executor:
            futures = []

            for start in range(first_page, last_page + 1, batch_size):
                end = min(start + batch_size - 1, last_page)
                future = executor.submit(convert_batch, start, end)
                futures.append((start, end, future))

            # Collect results as they complete
            for start, end, future in futures:
                try:
                    batch_images = future.result(
                        timeout=300)  # 5 minute timeout
                    all_images.extend(batch_images)
                    print(
                        f"   ✅ Completed pages {start}-{end} ({len(batch_images)} images)")
                except Exception as e:
                    print(f"   ❌ Failed pages {start}-{end}: {e}")

        return all_images

    def _memory_optimized_convert(self, pdf_path, dpi, output_folder, first_page,
                                  last_page, fmt, batch_size, use_process_pool, **kwargs):
        """Memory-optimized streaming conversion for large files"""

        page_count = PDFInfoExtractor.get_page_count(pdf_path)

        if first_page is None:
            first_page = 1
        if last_page is None:
            last_page = page_count

        # Force output to disk to save memory
        if output_folder is None:
            output_folder = tempfile.mkdtemp(prefix="pdf_optimizer_")
            auto_cleanup = True
        else:
            auto_cleanup = False

        kwargs.update({
            'output_folder': output_folder,
            'paths_only': True,  # Return paths instead of loaded images
            'thread_count': 1
        })

        if self.is_windows:
            kwargs.setdefault('use_pdftocairo', True)

        # Process one page at a time for maximum memory efficiency
        image_paths = []
        total_pages = last_page - first_page + 1

        print(f"💾 Memory-optimized processing: {total_pages} pages")
        print(f"📁 Output folder: {output_folder}")

        try:
            for page_num in range(first_page, last_page + 1):
                if page_num == first_page or page_num % 10 == 0:
                    progress = ((page_num - first_page + 1) /
                                total_pages) * 100
                    print(
                        f"   Processing page {page_num}/{last_page} ({progress:.1f}%)")

                try:
                    paths = original_convert_from_path(
                        pdf_path, dpi=dpi, first_page=page_num,
                        last_page=page_num, fmt=fmt, **kwargs
                    )
                    image_paths.extend(paths)

                except Exception as e:
                    print(f"   ❌ Failed page {page_num}: {e}")
                    continue

                # Aggressive garbage collection
                if page_num % 10 == 0:
                    import gc
                    gc.collect()

            print(f"✅ Completed: {len(image_paths)} files saved")
            return image_paths

        finally:
            if auto_cleanup:
                try:
                    # Don't clean up - return paths are in the temp directory
                    print(f"📁 Files saved in: {output_folder}")
                    print(
                        "   (Temporary directory - files will be cleaned up on system restart)")
                except:
                    pass


def convert_pdf_fast(pdf_path: Union[str, Path], **kwargs) -> List:
    """
    Quick conversion with automatic optimization

    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments (dpi, fmt, etc.)

    Returns:
        List of PIL Images or file paths

    Example:
        images = convert_pdf_fast("document.pdf", dpi=300, fmt="png")
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required. Install with: pip install pdf2image pillow")

    converter = OptimizedPDFConverter()
    return converter.convert_from_path_optimized(
        pdf_path, optimization_strategy="auto", **kwargs
    )


def convert_pdf_memory_efficient(pdf_path: Union[str, Path], **kwargs) -> List:
    """
    Memory-efficient conversion for large files

    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments

    Returns:
        List of file paths (saves to disk to conserve memory)

    Example:
        paths = convert_pdf_memory_efficient("large_file.pdf")
        print(f"Images saved to: {paths}")
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required. Install with: pip install pdf2image pillow")

    converter = OptimizedPDFConverter()
    return converter.convert_from_path_optimized(
        pdf_path, optimization_strategy="memory", **kwargs
    )


def convert_pdf_parallel(pdf_path: Union[str, Path], max_workers: int = None, **kwargs) -> List:
    """
    Parallel conversion (works better on Unix systems)

    Args:
        pdf_path: Path to PDF file
        max_workers: Maximum number of worker threads
        **kwargs: Additional arguments

    Returns:
        List of PIL Images or file paths

    Example:
        images = convert_pdf_parallel("document.pdf", max_workers=4)
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required. Install with: pip install pdf2image pillow")

    converter = OptimizedPDFConverter()

    if max_workers:
        # Calculate batch size based on worker count
        try:
            page_count = PDFInfoExtractor.get_page_count(str(pdf_path))
            kwargs['batch_size'] = max(1, page_count // max_workers)
        except:
            pass

    return converter.convert_from_path_optimized(
        pdf_path, optimization_strategy="async",
        use_process_pool=True, **kwargs
    )


def apply_patch():
    """
    Apply Windows optimization patch to pdf2image.convert_from_path

    After calling this, you can use pdf2image normally and it will be optimized:

    Example:
        apply_patch()
        from pdf2image import convert_from_path
        images = convert_from_path("file.pdf")  # Now optimized!
    """

    if not PDF2IMAGE_AVAILABLE:
        print("❌ pdf2image not available. Install with: pip install pdf2image pillow")
        return False

    try:
        import pdf2image

        @functools.wraps(original_convert_from_path)
        def optimized_wrapper(pdf_path, **kwargs):
            # Auto-optimize for Windows
            if platform.system() == "Windows":
                original_thread_count = kwargs.get('thread_count', None)

                if 'thread_count' not in kwargs:
                    kwargs['thread_count'] = 1
                elif kwargs.get('thread_count', 1) > 1:
                    print(
                        f"🪟 Windows optimization: reducing thread_count from {original_thread_count} to 1")
                    kwargs['thread_count'] = 1

                if 'use_pdftocairo' not in kwargs:
                    kwargs['use_pdftocairo'] = True
                    print("🪟 Windows optimization: enabled pdftocairo")

                # For large files, use batch processing
                try:
                    page_count = PDFInfoExtractor.get_page_count(str(pdf_path))
                    if page_count > 10:
                        print(
                            f"📦 Large file detected ({page_count} pages), using batch processing...")
                        converter = OptimizedPDFConverter()
                        return converter.convert_from_path_optimized(
                            pdf_path, optimization_strategy="batch", **kwargs
                        )
                except:
                    pass

            return original_convert_from_path(pdf_path, **kwargs)

        # Replace the function
        pdf2image.convert_from_path = optimized_wrapper

        print("✅ PDF optimization patch applied successfully!")
        if platform.system() == "Windows":
            print("   🪟 Windows optimizations enabled:")
            print("     - Single threading (thread_count=1)")
            print("     - pdftocairo enabled")
            print("     - Batch processing for large files")
        else:
            print(f"   🐧 {platform.system()} optimizations enabled")

        return True

    except Exception as e:
        print(f"❌ Failed to apply patch: {e}")
        return False


def benchmark_strategies(pdf_path: Union[str, Path], strategies: List[str] = None) -> Dict:
    """
    Benchmark different conversion strategies

    Args:
        pdf_path: Path to PDF file
        strategies: List of strategies to test (default: ["single", "batch"])

    Returns:
        Dictionary with benchmark results

    Example:
        results = benchmark_strategies("test.pdf")
        fastest = min(results, key=lambda x: results[x].get('time', float('inf')) if results[x]['success'] else float('inf'))
        print(f"Fastest: {fastest}")
    """

    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image is required for benchmarking")

    if strategies is None:
        strategies = ["single", "batch", "async", "memory"]

    converter = OptimizedPDFConverter()
    results = {}

    print(f"🏁 Benchmarking PDF conversion: {pdf_path}")
    print(f"🧪 Testing strategies: {', '.join(strategies)}\n")

    for strategy in strategies:
        print(f"Testing {strategy}...")
        start_time = time.time()

        try:
            images = converter.convert_from_path_optimized(
                pdf_path,
                optimization_strategy=strategy,
                dpi=150  # Lower DPI for faster benchmarking
            )

            end_time = time.time()
            duration = end_time - start_time

            results[strategy] = {
                'time': duration,
                'pages': len(images),
                'success': True,
                'pages_per_second': len(images) / duration if duration > 0 else 0
            }

            print(
                f"  ✅ {len(images)} pages in {duration:.2f}s ({len(images)/duration:.1f} pages/sec)\n")

        except Exception as e:
            results[strategy] = {
                'time': None,
                'error': str(e),
                'success': False,
                'pages_per_second': 0
            }
            print(f"  ❌ Failed: {e}\n")

    # Summary
    successful = {k: v for k, v in results.items() if v['success']}
    if successful:
        fastest = min(successful, key=lambda x: successful[x]['time'])
        print(
            f"🏆 Fastest strategy: {fastest} ({successful[fastest]['time']:.2f}s)")
        print(
            f"📊 Speed improvement vs slowest: {max(successful.values(), key=lambda x: x['time'])['time'] / successful[fastest]['time']:.1f}x")
    else:
        print("❌ No strategies succeeded")

    return results


def get_pdf_info(pdf_path: Union[str, Path]) -> Dict:
    """
    Get information about a PDF file

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with PDF information

    Example:
        info = get_pdf_info("document.pdf")
        print(f"Pages: {info['pages']}, Size: {info['size_mb']:.1f}MB")
    """
    pdf_path = str(pdf_path)

    try:
        pages = PDFInfoExtractor.get_page_count(pdf_path)
        size_bytes = os.path.getsize(pdf_path)
        size_mb = size_bytes / (1024 * 1024)

        return {
            'pages': pages,
            'size_bytes': size_bytes,
            'size_mb': size_mb,
            'path': pdf_path,
            'exists': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'path': pdf_path,
            'exists': os.path.exists(pdf_path)
        }


def main():
    """Example usage and testing"""

    print("🚀 PDF Conversion Optimizer - Standalone Version")
    print("=" * 60)

    # Check dependencies
    print("📋 Checking dependencies...")

    if PDF2IMAGE_AVAILABLE:
        print("  ✅ pdf2image available")
    else:
        print("  ❌ pdf2image not installed!")
        print("     Install with: pip install pdf2image")

    if PIL_AVAILABLE:
        print("  ✅ PIL/Pillow available")
    else:
        print("  ❌ PIL/Pillow not installed!")
        print("     Install with: pip install pillow")

    # System info
    print(f"\n💻 System: {platform.system()}")
    if platform.system() == "Windows":
        print("  🪟 Windows optimizations will be applied")
    print(f"💿 CPU cores: {os.cpu_count()}")

    if not PDF2IMAGE_AVAILABLE:
        print("\n❌ Cannot continue without pdf2image. Please install it first.")
        return

    # Usage examples
    print("\n" + "=" * 60)
    print("📖 USAGE EXAMPLES")
    print("=" * 60)

    examples = """
1. Quick conversion (auto-optimized):
   from pdf_optimizer import convert_pdf_fast
   images = convert_pdf_fast("document.pdf", dpi=200)

2. Memory-efficient (for large files):
   from pdf_optimizer import convert_pdf_memory_efficient  
   paths = convert_pdf_memory_efficient("large_file.pdf")
   # Returns file paths instead of loading images into memory

3. Parallel processing:
   from pdf_optimizer import convert_pdf_parallel
   images = convert_pdf_parallel("document.pdf", max_workers=4)

4. Apply patch to original pdf2image:
   from pdf_optimizer import apply_patch
   apply_patch()
   
   from pdf2image import convert_from_path
   images = convert_from_path("document.pdf")  # Now optimized!

5. Manual strategy selection:
   from pdf_optimizer import OptimizedPDFConverter
   converter = OptimizedPDFConverter()
   images = converter.convert_from_path_optimized(
       "document.pdf", 
       optimization_strategy="batch",
       batch_size=5,
       dpi=300
   )

6. Get PDF information:
   from pdf_optimizer import get_pdf_info
   info = get_pdf_info("document.pdf")
   print(f"Pages: {info['pages']}, Size: {info['size_mb']:.1f}MB")

7. Benchmark different strategies:
   from pdf_optimizer import benchmark_strategies
   results = benchmark_strategies("test.pdf")
"""

    print(examples)

    # Test with example files
    test_files = ["test.pdf", "sample.pdf", "document.pdf", "example.pdf"]
    found_file = None

    for test_file in test_files:
        if os.path.exists(test_file):
            found_file = test_file
            break

    if found_file:
        print(f"📄 Found test file: {found_file}")

        # Get file info
        info = get_pdf_info(found_file)
        if 'error' not in info:
            print(f"   📊 {info['pages']} pages, {info['size_mb']:.1f}MB")

            # Quick test
            print("\n🧪 Running quick test...")
            try:
                start_time = time.time()
                images = convert_pdf_fast(found_file, dpi=150, fmt="png")
                duration = time.time() - start_time
                print(f"✅ Converted {len(images)} pages in {duration:.2f}s")

                if len(images) > 0:
                    print(
                        f"   📏 First image size: {images[0].size if hasattr(images[0], 'size') else 'N/A'}")

            except Exception as e:
                print(f"❌ Test failed: {e}")
        else:
            print(f"❌ Error reading file: {info['error']}")
    else:
        print(f"📄 No test files found. Try placing a PDF file named 'test.pdf' in the current directory.")

    print(f"\n✨ Ready to optimize your PDF conversions!")


if __name__ == "__main__":
    # main()
    results = benchmark_strategies("test.pdf")
    print(results)
