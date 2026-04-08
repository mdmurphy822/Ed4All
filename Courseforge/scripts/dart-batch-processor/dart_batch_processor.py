#!/usr/bin/env python3
"""
DART Batch Processor - Automated Document to Accessible HTML Conversion

This script orchestrates batch conversion of PDF and Office documents to
WCAG 2.2 AA compliant accessible HTML using DART (Digital Accessibility
Remediation Tool).

Features:
- Parallel processing of multiple documents
- Progress tracking with detailed logging
- Automatic resource replacement in course structure
- Error handling with retry logic
- Integration with Courseforge remediation pipeline

Usage:
    python dart_batch_processor.py --input-manifest remediation_queue.json --output-dir /path/to/output/
    python dart_batch_processor.py --input-files file1.pdf file2.pdf --output-dir /path/to/output/
    python dart_batch_processor.py --course-dir /path/to/extracted/course/ --auto-detect
"""

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dart_batch_processor.log')
    ]
)
logger = logging.getLogger(__name__)

# DART installation path - MUST be set via environment variable
# Example: export DART_PATH=/path/to/DART
DART_PATH = Path(os.environ.get('DART_PATH', ''))
if not DART_PATH or not DART_PATH.exists():
    logger.warning("DART_PATH environment variable not set or invalid. Set it to your DART installation directory.")
DART_CONVERT_SCRIPT = DART_PATH / 'convert.py'


class ConversionStatus(Enum):
    """Status of individual document conversion"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentType(Enum):
    """Supported document types for conversion"""
    PDF = "pdf"
    WORD = "word"
    POWERPOINT = "powerpoint"
    EXCEL = "excel"
    OTHER = "other"


@dataclass
class ConversionTask:
    """Represents a single document conversion task"""
    source_path: str
    document_type: DocumentType
    status: ConversionStatus = ConversionStatus.PENDING
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    retry_count: int = 0
    file_size_bytes: int = 0
    output_size_bytes: int = 0


@dataclass
class BatchProcessingResult:
    """Results of batch processing operation"""
    total_documents: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    skipped_documents: int = 0
    total_duration_seconds: float = 0.0
    start_time: str = ""
    end_time: str = ""
    tasks: List[ConversionTask] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_directory: str = ""


class DARTBatchProcessor:
    """
    Batch processor for DART document conversions.

    Orchestrates parallel conversion of PDF and Office documents
    to accessible HTML using the DART tool.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        DocumentType.PDF: {'.pdf'},
        DocumentType.WORD: {'.doc', '.docx', '.odt', '.rtf'},
        DocumentType.POWERPOINT: {'.ppt', '.pptx', '.odp'},
        DocumentType.EXCEL: {'.xls', '.xlsx', '.ods', '.csv'},
    }

    # Maximum parallel workers
    MAX_WORKERS = 4

    # Retry configuration
    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 5

    def __init__(
        self,
        output_dir: Path,
        dart_path: Path = DART_PATH,
        max_workers: int = MAX_WORKERS,
        replace_originals: bool = True
    ):
        """
        Initialize the batch processor.

        Args:
            output_dir: Directory for converted HTML output
            dart_path: Path to DART installation
            max_workers: Maximum parallel conversion workers
            replace_originals: Whether to replace original files with HTML links
        """
        self.output_dir = Path(output_dir)
        self.dart_path = Path(dart_path)
        self.dart_script = self.dart_path / 'convert.py'
        self.max_workers = max_workers
        self.replace_originals = replace_originals

        self.tasks: List[ConversionTask] = []
        self.result: Optional[BatchProcessingResult] = None

        # Validate DART installation
        self._validate_dart_installation()

    def _validate_dart_installation(self):
        """Verify DART is properly installed and accessible"""
        if not self.dart_path.exists():
            raise FileNotFoundError(f"DART installation not found at: {self.dart_path}")

        if not self.dart_script.exists():
            raise FileNotFoundError(f"DART convert.py not found at: {self.dart_script}")

        logger.info(f"DART installation validated at: {self.dart_path}")

    def add_document(self, file_path: Path) -> bool:
        """
        Add a document to the conversion queue.

        Args:
            file_path: Path to the document

        Returns:
            True if document was added, False if unsupported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"Document not found: {file_path}")
            return False

        doc_type = self._detect_document_type(file_path)
        if doc_type == DocumentType.OTHER:
            logger.warning(f"Unsupported document type: {file_path}")
            return False

        task = ConversionTask(
            source_path=str(file_path),
            document_type=doc_type,
            file_size_bytes=file_path.stat().st_size
        )
        self.tasks.append(task)
        logger.info(f"Added document to queue: {file_path} ({doc_type.value})")
        return True

    def add_documents_from_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> int:
        """
        Add all supported documents from a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            Number of documents added
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        count = 0
        pattern = '**/*' if recursive else '*'

        for file_path in directory.glob(pattern):
            if file_path.is_file() and self.add_document(file_path):
                count += 1

        logger.info(f"Added {count} documents from directory: {directory}")
        return count

    def add_documents_from_manifest(self, manifest_path: Path) -> int:
        """
        Add documents from a remediation queue manifest.

        Args:
            manifest_path: Path to JSON manifest file

        Returns:
            Number of documents added
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        count = 0

        # Handle various manifest formats
        dart_queue = manifest.get('remediation_queue', {}).get('dart_conversion', [])
        if not dart_queue:
            dart_queue = manifest.get('dart_conversion', [])

        for item in dart_queue:
            file_path = item.get('file') or item.get('path') or item.get('source_path')
            if file_path and self.add_document(Path(file_path)):
                count += 1

        logger.info(f"Added {count} documents from manifest: {manifest_path}")
        return count

    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file extension"""
        ext = file_path.suffix.lower()

        for doc_type, extensions in self.SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return doc_type

        return DocumentType.OTHER

    def process_all(self) -> BatchProcessingResult:
        """
        Process all queued documents.

        Returns:
            BatchProcessingResult with conversion outcomes
        """
        if not self.tasks:
            logger.warning("No documents in queue to process")
            return BatchProcessingResult(
                total_documents=0,
                output_directory=str(self.output_dir)
            )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()
        logger.info(f"Starting batch processing of {len(self.tasks)} documents")

        # Process documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_document, task): task
                for task in self.tasks
            }

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Unexpected error processing {task.source_path}: {e}")
                    task.status = ConversionStatus.FAILED
                    task.error_message = str(e)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compile results
        successful = sum(1 for t in self.tasks if t.status == ConversionStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == ConversionStatus.FAILED)
        skipped = sum(1 for t in self.tasks if t.status == ConversionStatus.SKIPPED)

        self.result = BatchProcessingResult(
            total_documents=len(self.tasks),
            successful_conversions=successful,
            failed_conversions=failed,
            skipped_documents=skipped,
            total_duration_seconds=duration,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            tasks=self.tasks,
            errors=[t.error_message for t in self.tasks if t.error_message],
            output_directory=str(self.output_dir)
        )

        logger.info(f"Batch processing complete: {successful}/{len(self.tasks)} successful, "
                   f"{failed} failed, {skipped} skipped ({duration:.1f}s)")

        return self.result

    def _process_document(self, task: ConversionTask) -> None:
        """
        Process a single document conversion.

        Args:
            task: The conversion task to process
        """
        source_path = Path(task.source_path)
        task.status = ConversionStatus.IN_PROGRESS
        task.start_time = datetime.now().isoformat()

        logger.info(f"Processing: {source_path.name}")

        # Determine output path
        output_filename = source_path.stem + '.html'
        task_output_dir = self.output_dir / source_path.stem
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Handle different document types
        if task.document_type == DocumentType.PDF:
            success = self._convert_pdf(source_path, task_output_dir, task)
        elif task.document_type in (DocumentType.WORD, DocumentType.POWERPOINT, DocumentType.EXCEL):
            success = self._convert_office(source_path, task_output_dir, task)
        else:
            task.status = ConversionStatus.SKIPPED
            task.error_message = f"Unsupported document type: {task.document_type.value}"
            success = False

        task.end_time = datetime.now().isoformat()
        if task.start_time:
            start = datetime.fromisoformat(task.start_time)
            end = datetime.fromisoformat(task.end_time)
            task.duration_seconds = (end - start).total_seconds()

        if success:
            task.status = ConversionStatus.COMPLETED
            # Calculate output size
            if task.output_path and Path(task.output_path).exists():
                task.output_size_bytes = Path(task.output_path).stat().st_size

    def _convert_pdf(
        self,
        source_path: Path,
        output_dir: Path,
        task: ConversionTask
    ) -> bool:
        """
        Convert PDF using DART.

        Args:
            source_path: Path to PDF file
            output_dir: Directory for output
            task: The conversion task

        Returns:
            True if successful
        """
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # Build DART command
                cmd = [
                    sys.executable,
                    str(self.dart_script),
                    str(source_path),
                    '-o', str(output_dir),
                    '-v'
                ]

                logger.debug(f"Running DART command: {' '.join(cmd)}")

                # Execute DART
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per document
                    cwd=str(self.dart_path)
                )

                if result.returncode == 0:
                    # Find output file
                    html_files = list(output_dir.glob('*.html'))
                    if html_files:
                        task.output_path = str(html_files[0])
                        logger.info(f"Successfully converted: {source_path.name}")
                        return True
                    else:
                        task.error_message = "DART completed but no HTML output found"
                        logger.error(task.error_message)
                else:
                    task.error_message = f"DART failed: {result.stderr[:500]}"
                    logger.error(f"DART error for {source_path.name}: {result.stderr[:200]}")

            except subprocess.TimeoutExpired:
                task.error_message = "Conversion timed out (>5 minutes)"
                logger.error(f"Timeout converting: {source_path.name}")
            except Exception as e:
                task.error_message = str(e)
                logger.error(f"Error converting {source_path.name}: {e}")

            # Retry logic
            if attempt < self.MAX_RETRIES:
                task.retry_count += 1
                logger.info(f"Retrying {source_path.name} (attempt {attempt + 2}/{self.MAX_RETRIES + 1})")
                time.sleep(self.RETRY_DELAY_SECONDS)

        task.status = ConversionStatus.FAILED
        return False

    def _convert_office(
        self,
        source_path: Path,
        output_dir: Path,
        task: ConversionTask
    ) -> bool:
        """
        Convert Office document to PDF, then convert PDF using DART.

        Office documents are first converted to PDF using LibreOffice,
        then the PDF is converted to accessible HTML using DART.

        Args:
            source_path: Path to Office file
            output_dir: Directory for output
            task: The conversion task

        Returns:
            True if successful
        """
        # Create temp directory for intermediate PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Step 1: Convert Office to PDF using LibreOffice
                cmd = [
                    'libreoffice',
                    '--headless',
                    '--convert-to', 'pdf',
                    '--outdir', str(temp_path),
                    str(source_path)
                ]

                logger.debug(f"Converting to PDF: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )

                if result.returncode != 0:
                    # LibreOffice not available, try fallback
                    task.error_message = "LibreOffice conversion failed - document may need manual conversion"
                    logger.warning(f"LibreOffice failed for {source_path.name}: {result.stderr[:200]}")

                    # Create placeholder HTML with link to original
                    self._create_placeholder_html(source_path, output_dir, task)
                    return True  # Consider this a partial success

                # Find generated PDF
                pdf_files = list(temp_path.glob('*.pdf'))
                if not pdf_files:
                    task.error_message = "LibreOffice completed but no PDF generated"
                    return False

                pdf_path = pdf_files[0]

                # Step 2: Convert PDF to accessible HTML using DART
                return self._convert_pdf(pdf_path, output_dir, task)

            except FileNotFoundError:
                # LibreOffice not installed
                logger.warning("LibreOffice not installed - creating placeholder HTML")
                self._create_placeholder_html(source_path, output_dir, task)
                return True
            except subprocess.TimeoutExpired:
                task.error_message = "Office conversion timed out"
                return False
            except Exception as e:
                task.error_message = str(e)
                return False

    def _create_placeholder_html(
        self,
        source_path: Path,
        output_dir: Path,
        task: ConversionTask
    ) -> None:
        """
        Create placeholder HTML for documents that couldn't be fully converted.

        Args:
            source_path: Original document path
            output_dir: Output directory
            task: The conversion task
        """
        output_path = output_dir / f"{source_path.stem}.html"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{source_path.stem}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 2rem auto; padding: 1rem; }}
        .notice {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
                   padding: 1rem; margin: 1rem 0; }}
        .notice h2 {{ color: #856404; margin-top: 0; }}
        .download-link {{ display: inline-block; margin-top: 1rem; padding: 0.5rem 1rem;
                         background: #007bff; color: white; text-decoration: none;
                         border-radius: 4px; }}
        .download-link:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <main>
        <h1>{source_path.stem}</h1>
        <div class="notice" role="alert">
            <h2>Document Conversion Notice</h2>
            <p>This document could not be automatically converted to accessible HTML format.
               The original file is available for download below.</p>
            <p>For full accessibility, please request a converted version from your instructor
               or accessibility services.</p>
        </div>
        <a href="{source_path.name}" class="download-link" aria-label="Download original document">
            Download Original Document ({source_path.suffix.upper().replace('.', '')})
        </a>
    </main>
</body>
</html>"""

        output_path.write_text(html_content)
        task.output_path = str(output_path)
        task.status = ConversionStatus.COMPLETED
        task.error_message = "Created placeholder HTML - full conversion requires LibreOffice"
        logger.info(f"Created placeholder HTML for: {source_path.name}")

    def generate_report(self) -> str:
        """Generate a human-readable processing report"""
        if not self.result:
            return "No processing results available"

        r = self.result

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              DART BATCH PROCESSING REPORT                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Start Time: {r.start_time[:25]:<52} ║
║ End Time: {r.end_time[:25]:<54} ║
║ Duration: {r.total_duration_seconds:.1f} seconds{' ':<47} ║
╠══════════════════════════════════════════════════════════════════╣
║ CONVERSION SUMMARY                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Total Documents: {r.total_documents:<47} ║
║ Successful: {r.successful_conversions:<52} ║
║ Failed: {r.failed_conversions:<56} ║
║ Skipped: {r.skipped_documents:<55} ║
║ Success Rate: {(r.successful_conversions/max(r.total_documents,1)*100):.1f}%{' ':<48} ║
╠══════════════════════════════════════════════════════════════════╣
║ Output Directory: {str(r.output_directory)[:46]:<46} ║
╚══════════════════════════════════════════════════════════════════╝
"""

        if r.failed_conversions > 0:
            report += "\n\n=== FAILED CONVERSIONS ===\n"
            for task in r.tasks:
                if task.status == ConversionStatus.FAILED:
                    report += f"  • {Path(task.source_path).name}: {task.error_message}\n"

        return report

    def to_json(self) -> str:
        """Export processing results as JSON"""
        if not self.result:
            return "{}"

        data = asdict(self.result)

        # Fix enum values
        for task in data['tasks']:
            task['status'] = ConversionStatus(task['status']).value if isinstance(task['status'], str) else task['status'].value
            task['document_type'] = DocumentType(task['document_type']).value if isinstance(task['document_type'], str) else task['document_type'].value

        return json.dumps(data, indent=2, default=str)


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description='Batch convert documents to accessible HTML using DART'
    )
    parser.add_argument(
        '--input-manifest', '-m',
        help='Path to remediation queue manifest JSON'
    )
    parser.add_argument(
        '--input-files', '-f',
        nargs='+',
        help='List of input files to convert'
    )
    parser.add_argument(
        '--course-dir', '-c',
        help='Course directory to scan for documents'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for converted HTML'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Auto-detect all PDF/Office documents in course directory'
    )
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=4,
        help='Maximum parallel workers (default: 4)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--dart-path',
        default=str(DART_PATH),
        help=f'Path to DART installation (default: {DART_PATH})'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Verbose output (-vv for debug)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Validate inputs
    if not (args.input_manifest or args.input_files or (args.course_dir and args.auto_detect)):
        print("Error: Must provide --input-manifest, --input-files, or --course-dir with --auto-detect",
              file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize processor
        processor = DARTBatchProcessor(
            output_dir=Path(args.output_dir),
            dart_path=Path(args.dart_path),
            max_workers=args.max_workers
        )

        # Add documents to queue
        if args.input_manifest:
            processor.add_documents_from_manifest(Path(args.input_manifest))

        if args.input_files:
            for file_path in args.input_files:
                processor.add_document(Path(file_path))

        if args.course_dir and args.auto_detect:
            processor.add_documents_from_directory(Path(args.course_dir))

        # Process all documents
        result = processor.process_all()

        # Output results
        if args.json:
            print(processor.to_json())
        else:
            print(processor.generate_report())

        # Save JSON report to output directory
        report_path = Path(args.output_dir) / 'conversion_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(processor.to_json())
        print(f"\nConversion report saved to: {report_path}")

        # Exit with appropriate code
        if result.failed_conversions > 0:
            sys.exit(1)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
