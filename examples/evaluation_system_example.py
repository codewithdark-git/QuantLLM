#!/usr/bin/env python3
"""
Comprehensive Evaluation System Example

This example demonstrates how to use QuantLLM's evaluation and assessment tools
for comprehensive model evaluation on custom datasets with domain-specific metrics
and detailed reporting.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import quantllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantllm.evaluation import (
    ModelEvaluator,
    CustomDatasetEvaluator,
    DomainSpecificEvaluator,
    EvaluationConfig,
    StandardBenchmarks,
    DomainBenchmarks,
    CustomBenchmark,
    BenchmarkSuite,
    EvaluationReportGenerator,
    ComparisonReportGenerator,
    VisualizationGenerator,
    ReportConfig,
    create_custom_metric_suite,
    create_standard_benchmark_suite,
    create_domain_benchmark_suite
)

from quantllm.infrastructure.logging_system import setup_logging


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_data = [
        {
            "input": "What is the capital of France?",
            "target": "The capital of France is Paris.",
            "label": "geography"
        },
        {
            "input": "Explain photosynthesis in simple terms.",
            "target": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "label": "science"
        },
        {
            "input": "What are the symptoms of diabetes?",
            "target": "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
            "label": "medical"
        },
        {
            "input": "How do you calculate compound interest?",
            "target": "Compound interest is calculated using the formula: A = P(1 + r/n)^(nt), where A is the final amount, P is principal, r is annual interest rate, n is number of times interest is compounded per year, and t is time in years.",
            "label": "financial"
        },
        {
            "input": "Write a Python function to reverse a string.",
            "target": "def reverse_string(s):\n    return s[::-1]",
            "label": "code"
        }
    ]
    
    # Save to file
    dataset_path = "sample_evaluation_dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    return dataset_path


def demonstrate_basic_evaluation():
    """Demonstrate basic model evaluation."""
    print("=== Basic Model Evaluation ===")
    
    # Create evaluation configuration
    config = EvaluationConfig(
        metrics=["accuracy", "bleu", "rouge"],
        max_samples=5,
        batch_size=2,
        save_predictions=True
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    print(f"Available metrics: {evaluator.metrics_registry.list_metrics()}")
    print("Basic evaluation setup completed.")


def demonstrate_custom_dataset_evaluation():
    """Demonstrate evaluation on custom datasets."""
    print("\n=== Custom Dataset Evaluation ===")
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    print(f"Created sample dataset: {dataset_path}")
    
    # Create custom dataset evaluator
    config = EvaluationConfig(
        dataset_path=dataset_path,
        metrics=["accuracy", "bleu", "semantic_similarity"],
        max_samples=5,
        save_predictions=True
    )
    
    evaluator = CustomDatasetEvaluator(config)
    print("Custom dataset evaluator created.")
    
    # Clean up
    if os.path.exists(dataset_path):
        os.remove(dataset_path)


def demonstrate_domain_specific_evaluation():
    """Demonstrate domain-specific evaluation."""
    print("\n=== Domain-Specific Evaluation ===")
    
    domains = ["medical", "legal", "financial", "code", "scientific"]
    
    for domain in domains:
        print(f"\nDomain: {domain}")
        
        # Create domain-specific evaluator
        evaluator = DomainSpecificEvaluator(domain)
        
        # Show domain-specific metrics
        domain_metrics = evaluator.config.metrics
        print(f"  Metrics: {domain_metrics}")
        
        # Show available custom metrics
        custom_metrics = [name for name in evaluator.metrics_registry.list_metrics() 
                         if domain in name or "custom" in name]
        if custom_metrics:
            print(f"  Custom metrics: {custom_metrics}")


def demonstrate_benchmarks():
    """Demonstrate benchmark evaluation."""
    print("\n=== Benchmark Evaluation ===")
    
    # Show standard benchmarks
    print("Standard Benchmarks:")
    standard_benchmarks = StandardBenchmarks.list_benchmarks()
    for benchmark in standard_benchmarks[:5]:  # Show first 5
        config = StandardBenchmarks.get_benchmark(benchmark)
        print(f"  - {config.name}: {config.description}")
    
    # Show domain benchmarks
    print("\nDomain Benchmarks:")
    domain_benchmarks = DomainBenchmarks.list_benchmarks()
    for benchmark in domain_benchmarks[:5]:  # Show first 5
        config = DomainBenchmarks.get_benchmark(benchmark)
        print(f"  - {config.name} ({config.domain}): {config.description}")
    
    # Create benchmark suites
    print("\nBenchmark Suites:")
    
    # Standard benchmark suite for classification
    classification_suite = create_standard_benchmark_suite("classification")
    print(f"  Classification suite: {len(classification_suite.benchmarks)} benchmarks")
    
    # Domain benchmark suite for medical
    try:
        medical_suite = create_domain_benchmark_suite("medical")
        print(f"  Medical suite: {len(medical_suite.benchmarks)} benchmarks")
    except ValueError as e:
        print(f"  Medical suite: {e}")
    
    # Custom benchmark
    custom_benchmark = CustomBenchmark.from_dataset_path(
        name="Custom Test",
        dataset_path="test_data.json",
        metrics=["accuracy", "f1"],
        domain="general",
        description="Custom test dataset"
    )
    print(f"  Custom benchmark: {custom_benchmark.config.name}")


def demonstrate_custom_metrics():
    """Demonstrate custom metric creation."""
    print("\n=== Custom Metrics ===")
    
    # Show available metric suites
    domains = ["medical", "legal", "financial", "code"]
    tasks = ["dialogue", "summarization", "translation", "generation"]
    
    for domain in domains:
        for task in tasks:
            metrics = create_custom_metric_suite(domain, task)
            print(f"  {domain.title()} + {task.title()}: {len(metrics)} metrics")
            if len(metrics) <= 5:
                print(f"    Metrics: {metrics}")
            else:
                print(f"    Sample metrics: {metrics[:3]}...")


def demonstrate_report_generation():
    """Demonstrate report generation."""
    print("\n=== Report Generation ===")
    
    # Create report configurations
    html_config = ReportConfig(
        output_dir="evaluation_reports",
        format="html",
        include_plots=True,
        title="Model Evaluation Report"
    )
    
    json_config = ReportConfig(
        output_dir="evaluation_reports",
        format="json",
        include_detailed_metrics=True
    )
    
    markdown_config = ReportConfig(
        output_dir="evaluation_reports",
        format="markdown",
        include_predictions=False
    )
    
    print("Report configurations created:")
    print(f"  HTML config: {html_config.format} format, plots: {html_config.include_plots}")
    print(f"  JSON config: {json_config.format} format, detailed: {json_config.include_detailed_metrics}")
    print(f"  Markdown config: {markdown_config.format} format, predictions: {markdown_config.include_predictions}")
    
    # Create report generators
    html_generator = EvaluationReportGenerator(html_config)
    comparison_generator = ComparisonReportGenerator(html_config)
    viz_generator = VisualizationGenerator(html_config)
    
    print("\nReport generators created:")
    print(f"  Individual reports: {type(html_generator).__name__}")
    print(f"  Comparison reports: {type(comparison_generator).__name__}")
    print(f"  Visualizations: {type(viz_generator).__name__}")


def demonstrate_advanced_features():
    """Demonstrate advanced evaluation features."""
    print("\n=== Advanced Features ===")
    
    # Custom metric function
    def custom_length_metric(predictions, references, **kwargs):
        """Custom metric that measures average prediction length."""
        if not predictions:
            return {"avg_prediction_length": 0.0}
        
        lengths = [len(pred.split()) for pred in predictions]
        avg_length = sum(lengths) / len(lengths)
        return {"avg_prediction_length": avg_length}
    
    # Create evaluation config with custom metric
    config = EvaluationConfig(
        metrics=["accuracy", "bleu"],
        custom_metrics={"custom_length": custom_length_metric},
        max_samples=10,
        domain="general"
    )
    
    print("Custom metric configuration:")
    print(f"  Standard metrics: {config.metrics}")
    print(f"  Custom metrics: {list(config.custom_metrics.keys())}")
    
    # Domain-specific configurations
    medical_config = EvaluationConfig(
        metrics=["accuracy", "medical_terminology_coverage"],
        domain="medical",
        task_type="question-answering"
    )
    
    code_config = EvaluationConfig(
        metrics=["accuracy", "code_compilation_score"],
        domain="code",
        task_type="code-generation"
    )
    
    print("\nDomain-specific configurations:")
    print(f"  Medical: {medical_config.metrics}")
    print(f"  Code: {code_config.metrics}")


def demonstrate_integration_example():
    """Demonstrate a complete evaluation workflow."""
    print("\n=== Complete Evaluation Workflow ===")
    
    print("1. Dataset Preparation")
    dataset_path = create_sample_dataset()
    print(f"   Created dataset: {dataset_path}")
    
    print("2. Evaluation Configuration")
    config = EvaluationConfig(
        dataset_path=dataset_path,
        metrics=["accuracy", "bleu", "rouge", "semantic_similarity"],
        max_samples=5,
        batch_size=2,
        save_predictions=True,
        domain="general"
    )
    print(f"   Configured metrics: {config.metrics}")
    
    print("3. Evaluator Setup")
    evaluator = CustomDatasetEvaluator(config)
    print("   Custom dataset evaluator created")
    
    print("4. Report Configuration")
    report_config = ReportConfig(
        output_dir="evaluation_reports",
        format="html",
        include_plots=True,
        include_detailed_metrics=True,
        title="Comprehensive Evaluation Report"
    )
    print(f"   Report format: {report_config.format}")
    
    print("5. Report Generator Setup")
    report_generator = EvaluationReportGenerator(report_config)
    comparison_generator = ComparisonReportGenerator(report_config)
    viz_generator = VisualizationGenerator(report_config)
    print("   Report generators ready")
    
    print("6. Benchmark Suite")
    # Create a mixed benchmark suite
    benchmarks = ["glue_cola", "squad_v1"]  # Standard benchmarks
    try:
        benchmark_suite = BenchmarkSuite(benchmarks)
        print(f"   Benchmark suite with {len(benchmark_suite.benchmarks)} benchmarks")
    except Exception as e:
        print(f"   Benchmark suite creation: {e}")
    
    print("\n✓ Complete evaluation workflow configured successfully!")
    print("  Ready for model evaluation with:")
    print("  - Custom dataset evaluation")
    print("  - Domain-specific metrics")
    print("  - Comprehensive reporting")
    print("  - Benchmark comparisons")
    print("  - Visualization generation")
    
    # Clean up
    if os.path.exists(dataset_path):
        os.remove(dataset_path)


def main():
    """Main demonstration function."""
    print("QuantLLM Evaluation System Demonstration")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    try:
        # Run demonstrations
        demonstrate_basic_evaluation()
        demonstrate_custom_dataset_evaluation()
        demonstrate_domain_specific_evaluation()
        demonstrate_benchmarks()
        demonstrate_custom_metrics()
        demonstrate_report_generation()
        demonstrate_advanced_features()
        demonstrate_integration_example()
        
        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nThe evaluation system provides:")
        print("• Comprehensive model evaluation on custom datasets")
        print("• Domain-specific evaluation metrics and benchmarks")
        print("• Detailed report generation and comparison tools")
        print("• Flexible configuration and extensibility")
        print("• Integration with standard NLP benchmarks")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())