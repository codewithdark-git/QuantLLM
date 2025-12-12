#!/usr/bin/env python3
"""
Comprehensive Validation System Example

This example demonstrates how to use QuantLLM's comprehensive validation
and benchmarking system to validate quantized models, run regression tests,
and generate quality reports.
"""

import sys
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '.')

from quantllm.validation.validation_system import (
    ComprehensiveValidationSystem, 
    ValidationConfig,
    ComprehensiveValidationResult
)
from quantllm.engine.quality import QualityMetrics
from quantllm.engine.quality_assurance import (
    QualityThreshold, 
    RegressionTestCase
)
from quantllm.config.quantization_config import QuantizationConfig

def create_sample_validation_system():
    """Create a sample validation system with custom configuration."""
    print("Creating comprehensive validation system...")
    
    # Create custom validation configuration
    config = ValidationConfig(
        # Quality thresholds
        max_perplexity_increase=0.15,  # 15% max increase
        max_accuracy_drop=0.05,        # 5% max drop
        min_compression_ratio=1.5,     # Minimum 1.5x compression
        
        # Benchmarking settings
        num_inference_steps=50,
        num_warmup_steps=5,
        enable_quality_metrics=True,
        enable_hardware_monitoring=True,
        
        # Regression testing
        enable_regression_tests=True,
        regression_test_tolerance=0.1,
        
        # Trend analysis
        enable_trend_analysis=True,
        trend_window_days=30,
        
        # Reporting
        generate_reports=True,
        generate_visualizations=False,  # Disabled for this example
        save_results=True
    )
    
    # Create validation system
    with tempfile.TemporaryDirectory() as temp_dir:
        validation_system = ComprehensiveValidationSystem(
            config=config,
            output_dir=temp_dir
        )
        
        print(f"✓ Validation system created with output directory: {temp_dir}")
        return validation_system, temp_dir

def demonstrate_threshold_management(validation_system):
    """Demonstrate quality threshold management."""
    print("\n" + "="*50)
    print("QUALITY THRESHOLD MANAGEMENT")
    print("="*50)
    
    # Add custom thresholds
    custom_thresholds = [
        QualityThreshold(
            metric_name="model_size_mb",
            threshold_value=2000.0,  # Max 2GB
            comparison_type="max_value",
            severity="warning",
            description="Model size should be reasonable for deployment"
        ),
        QualityThreshold(
            metric_name="inference_latency_ms",
            threshold_value=100.0,  # Max 100ms
            comparison_type="max_value",
            severity="error",
            description="Inference should be fast enough for real-time use"
        ),
        QualityThreshold(
            metric_name="compression_ratio",
            threshold_value=2.0,  # Min 2x compression
            comparison_type="min_value",
            severity="warning",
            description="Should achieve meaningful compression"
        )
    ]
    
    for threshold in custom_thresholds:
        validation_system.threshold_manager.add_threshold(threshold)
        print(f"✓ Added threshold: {threshold.metric_name} ({threshold.comparison_type})")
    
    # Display all thresholds
    print(f"\nTotal thresholds configured: {len(validation_system.threshold_manager.thresholds)}")
    for name, threshold in validation_system.threshold_manager.thresholds.items():
        print(f"  - {name}: {threshold.threshold_value} ({threshold.severity})")

def demonstrate_regression_testing(validation_system):
    """Demonstrate regression test management."""
    print("\n" + "="*50)
    print("REGRESSION TEST MANAGEMENT")
    print("="*50)
    
    # Create sample regression test cases
    test_cases = [
        {
            "test_id": "llama_7b_4bit_quality",
            "description": "Quality regression test for LLaMA 7B 4-bit quantization",
            "expected_metrics": QualityMetrics(
                perplexity=5.2,
                accuracy=0.82,
                custom_metrics={
                    "model_size_mb": 3500.0,
                    "compression_ratio": 4.0
                }
            ),
            "tolerance": {
                "perplexity": 0.1,  # 10% tolerance
                "accuracy": 0.05,   # 5% tolerance
                "model_size_mb": 0.1,
                "compression_ratio": 0.2
            }
        },
        {
            "test_id": "gpt_3b_8bit_performance",
            "description": "Performance regression test for GPT 3B 8-bit quantization",
            "expected_metrics": QualityMetrics(
                perplexity=4.8,
                accuracy=0.85,
                custom_metrics={
                    "inference_latency_ms": 45.0,
                    "throughput_tokens_per_s": 120.0
                }
            ),
            "tolerance": {
                "perplexity": 0.08,
                "accuracy": 0.03,
                "inference_latency_ms": 0.15,
                "throughput_tokens_per_s": 0.1
            }
        }
    ]
    
    # Add regression tests
    for test_case_data in test_cases:
        validation_system.add_regression_test(
            test_id=test_case_data["test_id"],
            model_config={"model_type": "test_model"},
            quantization_config=QuantizationConfig(method="test", bits=4),
            expected_metrics=test_case_data["expected_metrics"],
            tolerance=test_case_data["tolerance"]
        )
        print(f"✓ Added regression test: {test_case_data['test_id']}")
    
    print(f"\nTotal regression tests: {len(validation_system.regression_suite.test_cases)}")

def demonstrate_quality_validation():
    """Demonstrate quality validation capabilities."""
    print("\n" + "="*50)
    print("QUALITY VALIDATION DEMONSTRATION")
    print("="*50)
    
    # Create sample quality metrics
    original_metrics = QualityMetrics(
        perplexity=4.5,
        accuracy=0.87,
        bleu_score=0.72,
        custom_metrics={
            "model_size_mb": 6800.0,
            "inference_latency_ms": 35.0,
            "compression_ratio": 1.0  # Original model
        }
    )
    
    quantized_metrics = QualityMetrics(
        perplexity=5.1,  # 13% increase
        accuracy=0.84,   # 3.4% decrease
        bleu_score=0.69, # 4.2% decrease
        custom_metrics={
            "model_size_mb": 1700.0,  # 4x compression
            "inference_latency_ms": 28.0,  # 20% faster
            "compression_ratio": 4.0
        }
    )
    
    print("Original metrics:")
    print(f"  Perplexity: {original_metrics.perplexity}")
    print(f"  Accuracy: {original_metrics.accuracy:.3f}")
    print(f"  BLEU Score: {original_metrics.bleu_score:.3f}")
    print(f"  Model Size: {original_metrics.custom_metrics['model_size_mb']:.0f} MB")
    
    print("\nQuantized metrics:")
    print(f"  Perplexity: {quantized_metrics.perplexity} (+{((quantized_metrics.perplexity/original_metrics.perplexity)-1)*100:.1f}%)")
    print(f"  Accuracy: {quantized_metrics.accuracy:.3f} ({((quantized_metrics.accuracy/original_metrics.accuracy)-1)*100:.1f}%)")
    print(f"  BLEU Score: {quantized_metrics.bleu_score:.3f} ({((quantized_metrics.bleu_score/original_metrics.bleu_score)-1)*100:.1f}%)")
    print(f"  Model Size: {quantized_metrics.custom_metrics['model_size_mb']:.0f} MB ({quantized_metrics.custom_metrics['compression_ratio']:.1f}x compression)")
    print(f"  Latency: {quantized_metrics.custom_metrics['inference_latency_ms']:.0f} ms ({((quantized_metrics.custom_metrics['inference_latency_ms']/original_metrics.custom_metrics['inference_latency_ms'])-1)*100:.1f}%)")
    
    return original_metrics, quantized_metrics

def demonstrate_benchmarking_comparison():
    """Demonstrate benchmarking and method comparison."""
    print("\n" + "="*50)
    print("BENCHMARKING AND METHOD COMPARISON")
    print("="*50)
    
    # Sample benchmark results for different quantization methods
    benchmark_results = {
        "GGUF_Q4_K_M": {
            "compression_ratio": 3.8,
            "inference_latency_ms": 32.0,
            "model_size_mb": 1800.0,
            "perplexity": 5.2,
            "accuracy": 0.83
        },
        "GPTQ_4bit": {
            "compression_ratio": 4.1,
            "inference_latency_ms": 28.0,
            "model_size_mb": 1650.0,
            "perplexity": 5.0,
            "accuracy": 0.84
        },
        "AWQ_4bit": {
            "compression_ratio": 3.9,
            "inference_latency_ms": 30.0,
            "model_size_mb": 1750.0,
            "perplexity": 4.9,
            "accuracy": 0.85
        }
    }
    
    print("Method Comparison Results:")
    print("-" * 80)
    print(f"{'Method':<12} {'Compression':<12} {'Latency(ms)':<12} {'Size(MB)':<10} {'Perplexity':<11} {'Accuracy'}")
    print("-" * 80)
    
    for method, results in benchmark_results.items():
        print(f"{method:<12} {results['compression_ratio']:<12.1f} {results['inference_latency_ms']:<12.1f} "
              f"{results['model_size_mb']:<10.0f} {results['perplexity']:<11.1f} {results['accuracy']:.3f}")
    
    # Determine best methods
    best_compression = max(benchmark_results.items(), key=lambda x: x[1]['compression_ratio'])
    best_speed = min(benchmark_results.items(), key=lambda x: x[1]['inference_latency_ms'])
    best_quality = min(benchmark_results.items(), key=lambda x: x[1]['perplexity'])
    
    print("\nRecommendations:")
    print(f"🚀 Best Speed: {best_speed[0]} ({best_speed[1]['inference_latency_ms']:.1f} ms)")
    print(f"📦 Best Compression: {best_compression[0]} ({best_compression[1]['compression_ratio']:.1f}x)")
    print(f"🎯 Best Quality: {best_quality[0]} (perplexity: {best_quality[1]['perplexity']:.1f})")

def demonstrate_reporting():
    """Demonstrate report generation capabilities."""
    print("\n" + "="*50)
    print("REPORT GENERATION")
    print("="*50)
    
    print("The validation system can generate comprehensive reports including:")
    print("✓ Quality assessment reports (JSON format)")
    print("✓ Benchmark comparison reports")
    print("✓ Regression test results")
    print("✓ Trend analysis reports")
    print("✓ Interactive visualizations (when matplotlib/plotly available)")
    print("✓ Executive summary reports")
    
    print("\nReport types available:")
    report_types = [
        "Quality Dashboard - Interactive overview of model quality",
        "Trend Analysis - Quality metrics over time",
        "Benchmark Comparison - Side-by-side method comparison",
        "Regression Report - Test results and failures",
        "Executive Summary - High-level findings and recommendations"
    ]
    
    for i, report_type in enumerate(report_types, 1):
        print(f"  {i}. {report_type}")

def main():
    """Main demonstration function."""
    print("=" * 60)
    print("QUANTLLM COMPREHENSIVE VALIDATION SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Create validation system
        validation_system, temp_dir = create_sample_validation_system()
        
        # Demonstrate different capabilities
        demonstrate_threshold_management(validation_system)
        demonstrate_regression_testing(validation_system)
        original_metrics, quantized_metrics = demonstrate_quality_validation()
        demonstrate_benchmarking_comparison()
        demonstrate_reporting()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("The Comprehensive Validation System provides:")
        print("✅ Automated quality validation with configurable thresholds")
        print("✅ Regression testing for model quality preservation")
        print("✅ Performance benchmarking and method comparison")
        print("✅ Quality trend analysis and monitoring")
        print("✅ Comprehensive reporting and visualization")
        print("✅ Integration with all QuantLLM quantization backends")
        
        print(f"\n📁 Example reports would be saved to: {temp_dir}")
        print("🎉 Validation system demonstration completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)