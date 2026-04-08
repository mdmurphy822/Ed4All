#!/usr/bin/env python3
"""
HTML Generator - Convert structured course content into Bootstrap HTML pages

This script implements atomic operations and comprehensive validation to generate
professional HTML pages with Bootstrap framework and accessibility compliance.

Author: Claude Code Assistant
Version: 1.0.0
Created: 2025-08-05
"""

import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class HTMLGenerator:
    """
    Generates HTML pages from structured course content JSON.
    
    Implements atomic operations with comprehensive validation to create
    professional Bootstrap-based HTML pages for IMSCC generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize HTML generator with configuration and logging.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.temp_files = []
        self.execution_lock = None
        
    def setup_logging(self):
        """Configure comprehensive logging for all operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('html_generator.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load HTML generator configuration with defaults.
        
        Args:
            config_path (str, optional): Path to config file
            
        Returns:
            dict: Configuration parameters
        """
        default_config = {
            "bootstrap_version": "4.3.1",
            "css_framework": {
                "cdn_primary": "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css",
                "cdn_fallback": "https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.3.1/css/bootstrap.min.css"
            },
            "javascript_framework": {
                "bootstrap_js": "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js",
                "jquery": "https://code.jquery.com/jquery-3.3.1.slim.min.js",
                "font_awesome": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
            },
            "accessibility": {
                "wcag_level": "AA",
                "keyboard_navigation": True,
                "screen_reader_support": True,
                "high_contrast_mode": True
            },
            "styling": {
                "line_height": 1.6,
                "font_family": "Arial, sans-serif",
                "container_max_width": "1200px",
                "accordion_animation_duration": "0.3s"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def validate_execution_environment(self, input_path: str, output_dir: str):
        """
        Comprehensive pre-flight validation before HTML generation.
        
        Args:
            input_path (str): Path to structured JSON file
            output_dir (str): Output directory for HTML files
            
        Raises:
            SystemExit: If validation fails
        """
        self.logger.info("Starting execution environment validation")
        
        # Validate input file exists and is readable
        input_file = Path(input_path)
        if not input_file.exists():
            raise SystemExit(f"CRITICAL ERROR: Input file does not exist: {input_path}")
            
        if not input_file.is_file():
            raise SystemExit(f"CRITICAL ERROR: Input path is not a file: {input_path}")
        
        # Validate JSON structure
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.validate_input_structure(data)
        except json.JSONDecodeError as e:
            raise SystemExit(f"CRITICAL ERROR: Invalid JSON format: {e}")
        except Exception as e:
            raise SystemExit(f"CRITICAL ERROR: Cannot read input file: {e}")
        
        # Validate output directory
        output_path = Path(output_dir)
        if output_path.exists() and any(output_path.iterdir()):
            raise SystemExit(f"COLLISION DETECTED: Output directory not empty: {output_dir}")
        
        # Create execution lock
        self.execution_lock = output_path / '.html_generation_lock'
        if self.execution_lock.exists():
            raise SystemExit("EXECUTION LOCK ERROR: Another HTML generator process is running")
        
        # Create output directory and lock
        output_path.mkdir(parents=True, exist_ok=True)
        self.execution_lock.touch()
        
        self.logger.info("Execution environment validation passed")
    
    def validate_input_structure(self, data: Dict[str, Any]):
        """
        Validate structured JSON input meets requirements.
        
        Args:
            data (dict): Structured course data
            
        Raises:
            SystemExit: If structure validation fails
        """
        required_sections = ['course_info', 'weeks']
        for section in required_sections:
            if section not in data:
                raise SystemExit(f"VALIDATION ERROR: Missing required section: {section}")
        
        # Validate weeks structure
        if not isinstance(data['weeks'], list) or not data['weeks']:
            raise SystemExit("VALIDATION ERROR: Weeks section must be non-empty list")
        
        for week in data['weeks']:
            if 'sub_modules' not in week or not isinstance(week['sub_modules'], list):
                raise SystemExit(f"VALIDATION ERROR: Week {week.get('week_number', '?')} missing sub_modules")
            
            if len(week['sub_modules']) != 7:
                raise SystemExit(f"VALIDATION ERROR: Week {week.get('week_number', '?')} has {len(week['sub_modules'])} sub-modules, expected 7")
    
    def generate_html_template(self, title: str, content: str, page_type: str = "standard") -> str:
        """
        Generate complete HTML page with Bootstrap framework.
        
        Args:
            title (str): Page title
            content (str): Main page content
            page_type (str): Type of page for specific styling
            
        Returns:
            str: Complete HTML document
        """
        # Custom CSS for enhanced styling
        custom_css = f"""
        <style>
            body {{
                font-family: {self.config['styling']['font_family']};
                line-height: {self.config['styling']['line_height']};
            }}
            .container {{
                max-width: {self.config['styling']['container_max_width']};
            }}
            .content-paragraph {{
                margin-bottom: 1.5rem;
                text-align: justify;
            }}
            .content-section {{
                margin-bottom: 2rem;
            }}
            .accordion .card-header {{
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            .accordion .btn-link {{
                color: #495057;
                text-decoration: none;
                font-weight: 500;
            }}
            .accordion .btn-link:hover {{
                color: #007bff;
                text-decoration: none;
            }}
            .rotate-icon {{
                transition: transform {self.config['styling']['accordion_animation_duration']} ease;
            }}
            .rotate-icon.rotated {{
                transform: rotate(90deg);
            }}
            .key-concept-definition {{
                font-size: 1rem;
                color: #495057;
                line-height: 1.6;
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 15px;
                }}
                h1 {{
                    font-size: 1.75rem;
                }}
            }}
        </style>
        """
        
        # Custom JavaScript for accordion functionality
        custom_js = """
        <script>
            $(document).ready(function() {
                // Accordion icon rotation
                $('.accordion .btn-link').on('click', function() {
                    var icon = $(this).find('.rotate-icon');
                    setTimeout(function() {
                        if (icon.closest('.btn-link').attr('aria-expanded') === 'true') {
                            icon.addClass('rotated');
                        } else {
                            icon.removeClass('rotated');
                        }
                    }, 50);
                });
                
                // Expand/Collapse all functionality for accordions
                if ($('.accordion').length > 0) {
                    var expandAllBtn = '<div class="mb-3"><button class="btn btn-outline-primary btn-sm" id="expandAll">Expand All</button> <button class="btn btn-outline-secondary btn-sm" id="collapseAll">Collapse All</button></div>';
                    $('.accordion').before(expandAllBtn);
                    
                    $('#expandAll').on('click', function() {
                        $('.accordion .collapse').collapse('show');
                        $('.rotate-icon').addClass('rotated');
                    });
                    
                    $('#collapseAll').on('click', function() {
                        $('.accordion .collapse').collapse('hide');
                        $('.rotate-icon').removeClass('rotated');
                    });
                }
                
                // Keyboard navigation for accordion
                $('.accordion .btn-link').on('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        $(this).click();
                    }
                });
            });
        </script>
        """
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Linear Algebra Course - {title}">
    <title>{title}</title>
    
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="{self.config['css_framework']['cdn_primary']}" 
          onerror="this.onerror=null;this.href='{self.config['css_framework']['cdn_fallback']}';">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="{self.config['javascript_framework']['font_awesome']}">
    
    {custom_css}
</head>
<body>
    <div class="container mt-4">
        <header>
            <h1 class="mb-4">{title}</h1>
        </header>
        
        <main role="main">
            {content}
        </main>
        
        <footer class="mt-5 pt-4 border-top">
            <p class="text-muted text-center">
                <small>Linear Algebra: Foundations and Applications - Course Content</small>
            </p>
        </footer>
    </div>
    
    <!-- JavaScript Dependencies -->
    <script src="{self.config['javascript_framework']['jquery']}" 
            onerror="console.warn('jQuery failed to load from CDN')"></script>
    <script src="{self.config['javascript_framework']['bootstrap_js']}" 
            onerror="console.warn('Bootstrap JS failed to load from CDN')"></script>
    
    {custom_js}
</body>
</html>"""
        
        return html_template
    
    def format_content_paragraphs(self, content: str) -> str:
        """
        Format content text into properly structured paragraphs.
        
        Args:
            content (str): Raw content text
            
        Returns:
            str: HTML formatted content with proper paragraph structure
        """
        if not content:
            return '<p class="content-paragraph">Content to be developed.</p>'
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return '<p class="content-paragraph">Content to be developed.</p>'
        
        # Format each paragraph
        formatted_paragraphs = []
        for paragraph in paragraphs:
            # Clean paragraph text
            clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
            
            # Skip very short paragraphs
            if len(clean_paragraph.split()) < 10:
                continue
            
            # Ensure proper paragraph length (50-300 words as per standards)
            words = clean_paragraph.split()
            if len(words) > 300:
                # Split long paragraphs
                mid_point = len(words) // 2
                part1 = ' '.join(words[:mid_point])
                part2 = ' '.join(words[mid_point:])
                formatted_paragraphs.append(f'<p class="content-paragraph">{part1}</p>')
                formatted_paragraphs.append(f'<p class="content-paragraph">{part2}</p>')
            elif len(words) >= 50:
                formatted_paragraphs.append(f'<p class="content-paragraph">{clean_paragraph}</p>')
            else:
                # Combine short paragraphs
                if formatted_paragraphs:
                    last_p = formatted_paragraphs[-1]
                    combined = last_p.replace('</p>', f' {clean_paragraph}</p>')
                    formatted_paragraphs[-1] = combined
                else:
                    formatted_paragraphs.append(f'<p class="content-paragraph">{clean_paragraph}</p>')
        
        return '\n'.join(formatted_paragraphs) if formatted_paragraphs else '<p class="content-paragraph">Content to be developed.</p>'
    
    def generate_overview_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for module overview pages.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for overview page
        """
        content_html = f"""
        <div class="content-section">
            <h2>Learning Objectives</h2>
            {self.format_learning_objectives(sub_module.get('learning_objectives', []))}
        </div>
        
        <div class="content-section">
            <h2>Module Introduction</h2>
            {self.format_content_paragraphs(sub_module.get('content', ''))}
        </div>
        
        <div class="content-section">
            <h2>What You'll Learn This Week</h2>
            <div class="alert alert-info" role="alert">
                <i class="fas fa-info-circle" aria-hidden="true"></i>
                This module provides the foundation for understanding key concepts that will be explored in depth throughout the week.
            </div>
        </div>
        """
        
        return content_html
    
    def generate_concept_summary_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for concept summary pages.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for concept summary page
        """
        content_html = f"""
        <div class="content-section">
            <div class="row">
                <div class="col-md-12">
                    {self.format_content_paragraphs(sub_module.get('content', ''))}
                </div>
            </div>
        </div>
        
        {self.format_key_concepts_inline(sub_module.get('key_concepts', []))}
        
        <div class="content-section">
            <div class="alert alert-success" role="alert">
                <i class="fas fa-lightbulb" aria-hidden="true"></i>
                <strong>Key Insight:</strong> Understanding these fundamental concepts will prepare you for more advanced applications in upcoming modules.
            </div>
        </div>
        """
        
        return content_html
    
    def generate_key_concepts_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for interactive key concepts accordion.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for key concepts accordion
        """
        key_concepts = sub_module.get('key_concepts', [])
        
        if not key_concepts:
            # Extract concepts from content if not explicitly provided
            content = sub_module.get('content', '')
            key_concepts = self.extract_concepts_from_content(content)
        
        accordion_items = []
        for i, concept in enumerate(key_concepts[:10]):  # Maximum 10 concepts
            concept_id = f"concept{week_number}_{i+1}"
            
            if isinstance(concept, dict):
                term = concept.get('term', f'Concept {i+1}')
                definition = concept.get('definition', 'Definition to be provided.')
            else:
                # Handle string format
                term = f"Key Concept {i+1}"
                definition = str(concept)
            
            accordion_item = f"""
            <div class="card">
                <div class="card-header" id="heading{concept_id}">
                    <h2 class="mb-0">
                        <button class="btn btn-link btn-block text-left" type="button" 
                                data-toggle="collapse" data-target="#collapse{concept_id}" 
                                aria-expanded="false" aria-controls="collapse{concept_id}">
                            <i class="fas fa-chevron-right rotate-icon" aria-hidden="true"></i>
                            <span class="ml-2">{term}</span>
                        </button>
                    </h2>
                </div>
                <div id="collapse{concept_id}" class="collapse" 
                     aria-labelledby="heading{concept_id}" data-parent="#keyConceptsAccordion">
                    <div class="card-body">
                        <p class="key-concept-definition">{definition}</p>
                    </div>
                </div>
            </div>
            """
            accordion_items.append(accordion_item)
        
        content_html = f"""
        <div class="content-section">
            <p class="lead">Explore the key concepts for this module. Click on each term to reveal its definition and explanation.</p>
        </div>
        
        <div class="accordion" id="keyConceptsAccordion" role="tablist" aria-label="Key Concepts">
            {''.join(accordion_items)}
        </div>
        
        <div class="content-section mt-4">
            <div class="alert alert-primary" role="alert">
                <i class="fas fa-graduation-cap" aria-hidden="true"></i>
                <strong>Study Tip:</strong> Review these key concepts regularly and try to explain each one in your own words.
            </div>
        </div>
        """
        
        return content_html
    
    def generate_visual_content_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for visual/graphical/math display pages.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for visual content page
        """
        content_html = f"""
        <div class="content-section">
            <div class="row">
                <div class="col-md-12">
                    {self.format_content_paragraphs(sub_module.get('content', ''))}
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <h2>Visual Learning Elements</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-chart-line" aria-hidden="true"></i> Graphical Representations</h5>
                            <p class="card-text">Interactive visualizations and diagrams help illustrate complex mathematical concepts.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-calculator" aria-hidden="true"></i> Mathematical Notation</h5>
                            <p class="card-text">Properly formatted equations and mathematical expressions for clarity.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <div class="alert alert-info" role="alert">
                <i class="fas fa-eye" aria-hidden="true"></i>
                <strong>Visual Learning:</strong> Take time to study each diagram and graph carefully. Visual representations often make abstract concepts more concrete.
            </div>
        </div>
        """
        
        return content_html
    
    def generate_application_examples_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for application examples pages.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for application examples page
        """
        content_html = f"""
        <div class="content-section">
            <h2>Learning Concepts in Practice</h2>
            {self.format_content_paragraphs(sub_module.get('content', ''))}
        </div>
        
        <div class="content-section">
            <h2>Step-by-Step Examples</h2>
            <div class="card-deck">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <i class="fas fa-play" aria-hidden="true"></i> Example 1
                    </div>
                    <div class="card-body">
                        <p class="card-text">Detailed step-by-step demonstration showing how theoretical concepts apply to practical problems.</p>
                    </div>
                </div>
                <div class="card border-success">
                    <div class="card-header bg-success text-white">
                        <i class="fas fa-play" aria-hidden="true"></i> Example 2
                    </div>
                    <div class="card-body">
                        <p class="card-text">Additional examples showing different approaches and solution methods.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <div class="alert alert-warning" role="alert">
                <i class="fas fa-tools" aria-hidden="true"></i>
                <strong>Practice Opportunity:</strong> Try working through similar problems on your own to reinforce your understanding.
            </div>
        </div>
        """
        
        return content_html
    
    def generate_real_world_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for real world applications pages.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for real world applications page
        """
        content_html = f"""
        <div class="content-section">
            <h2>Real-World Applications</h2>
            {self.format_content_paragraphs(sub_module.get('content', ''))}
        </div>
        
        <div class="content-section">
            <h2>Industry Connections</h2>
            <div class="row">
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-laptop-code" aria-hidden="true"></i> Technology</h5>
                            <p class="card-text">Computer graphics, machine learning, and data analysis applications.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-cogs" aria-hidden="true"></i> Engineering</h5>
                            <p class="card-text">Structural analysis, control systems, and optimization problems.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-chart-bar" aria-hidden="true"></i> Economics</h5>
                            <p class="card-text">Economic modeling, market analysis, and financial optimization.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <div class="alert alert-success" role="alert">
                <i class="fas fa-briefcase" aria-hidden="true"></i>
                <strong>Career Relevance:</strong> These applications demonstrate why linear algebra is essential in many professional fields.
            </div>
        </div>
        """
        
        return content_html
    
    def generate_study_questions_html(self, sub_module: Dict[str, Any], week_number: int) -> str:
        """
        Generate HTML content for study questions pages.
        
        Args:
            sub_module (dict): Sub-module data
            week_number (int): Week number
            
        Returns:
            str: HTML content for study questions page
        """
        content_html = f"""
        <div class="content-section">
            <h2>Reflection and Review</h2>
            {self.format_content_paragraphs(sub_module.get('content', ''))}
        </div>
        
        <div class="content-section">
            <h2>Study Questions</h2>
            <div class="list-group">
                <div class="list-group-item">
                    <h5 class="mb-1"><i class="fas fa-question-circle text-primary" aria-hidden="true"></i> Knowledge Check</h5>
                    <p class="mb-1">What are the key concepts you learned in this module?</p>
                    <small class="text-muted">Consider the main ideas and their relationships.</small>
                </div>
                <div class="list-group-item">
                    <h5 class="mb-1"><i class="fas fa-lightbulb text-warning" aria-hidden="true"></i> Critical Thinking</h5>
                    <p class="mb-1">How do these concepts connect to what you already know?</p>
                    <small class="text-muted">Think about connections to previous modules or your experience.</small>
                </div>
                <div class="list-group-item">
                    <h5 class="mb-1"><i class="fas fa-rocket text-success" aria-hidden="true"></i> Application</h5>
                    <p class="mb-1">Where might you apply these concepts in your field?</p>
                    <small class="text-muted">Consider practical applications in your area of study or work.</small>
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <div class="alert alert-primary" role="alert">
                <i class="fas fa-pencil-alt" aria-hidden="true"></i>
                <strong>Study Strategy:</strong> Write brief answers to these questions to help consolidate your learning.
            </div>
        </div>
        """
        
        return content_html
    
    def format_learning_objectives(self, objectives: List[str]) -> str:
        """
        Format learning objectives into HTML list.
        
        Args:
            objectives (list): List of learning objectives
            
        Returns:
            str: HTML formatted objectives list
        """
        if not objectives:
            return '<p class="content-paragraph">Learning objectives will be provided.</p>'
        
        objectives_html = '<ul class="list-group list-group-flush">'
        for i, objective in enumerate(objectives):
            objectives_html += f'''
            <li class="list-group-item d-flex align-items-start">
                <i class="fas fa-check-circle text-success mt-1 mr-3" aria-hidden="true"></i>
                <span>{objective}</span>
            </li>
            '''
        objectives_html += '</ul>'
        
        return objectives_html
    
    def format_key_concepts_inline(self, key_concepts: List[Any]) -> str:
        """
        Format key concepts inline with content.
        
        Args:
            key_concepts (list): List of key concepts
            
        Returns:
            str: HTML formatted key concepts
        """
        if not key_concepts:
            return ''
        
        concepts_html = '''
        <div class="content-section">
            <h3>Key Terms</h3>
            <div class="row">
        '''
        
        for i, concept in enumerate(key_concepts[:6]):  # Limit to 6 inline concepts
            if isinstance(concept, dict):
                term = concept.get('term', f'Term {i+1}')
                definition = concept.get('definition', 'Definition provided.')
            else:
                term = f"Key Term {i+1}"
                definition = str(concept)
            
            concepts_html += f'''
            <div class="col-md-6 mb-3">
                <div class="card border-info">
                    <div class="card-body">
                        <h6 class="card-title text-info">{term}</h6>
                        <p class="card-text small">{definition[:100]}{'...' if len(definition) > 100 else ''}</p>
                    </div>
                </div>
            </div>
            '''
        
        concepts_html += '''
            </div>
        </div>
        '''
        
        return concepts_html
    
    def extract_concepts_from_content(self, content: str) -> List[Dict[str, str]]:
        """
        Extract key concepts from content when not explicitly provided.
        
        Args:
            content (str): Content text to extract concepts from
            
        Returns:
            list: List of extracted concept dictionaries
        """
        concepts = []
        
        # Look for definition patterns
        definition_patterns = [
            r'([A-Z][a-zA-Z\s]+):\s*([^.!?]+[.!?])',  # Term: Definition
            r'\*\*([^*]+)\*\*[:\s]*([^.!?]+[.!?])',   # **Term**: Definition
            r'_([^_]+)_[:\s]*([^.!?]+[.!?])'          # _Term_: Definition
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, content)
            for term, definition in matches[:5]:  # Limit to 5 extracted concepts
                concepts.append({
                    'term': term.strip(),
                    'definition': definition.strip()
                })
        
        # If no concepts found, create placeholder
        if not concepts:
            concepts = [
                {'term': 'Key Concept 1', 'definition': 'Important concept for this module.'},
                {'term': 'Key Concept 2', 'definition': 'Essential understanding for course progression.'},
                {'term': 'Key Concept 3', 'definition': 'Fundamental principle to master.'}
            ]
        
        return concepts
    
    def generate_html_files(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Generate all HTML files with atomic execution.
        
        Args:
            input_path (str): Path to structured JSON file
            output_dir (str): Output directory for HTML files
            
        Returns:
            dict: Generation results summary
        """
        try:
            # Pre-flight validation
            self.validate_execution_environment(input_path, output_dir)
            
            # Load structured data
            with open(input_path, 'r', encoding='utf-8') as f:
                course_data = json.load(f)
            
            output_path = Path(output_dir)
            generated_files = []
            
            # Generate HTML files for each week and sub-module
            for week in course_data['weeks']:
                week_number = week['week_number']
                
                for sub_module in week['sub_modules']:
                    module_type = sub_module['type']
                    
                    # Generate appropriate filename
                    if module_type == 'concept_summary':
                        # Number concept summaries
                        existing_summaries = len([f for f in generated_files if f'week_{week_number:02d}_concept_summary' in f])
                        filename = f"week_{week_number:02d}_concept_summary_{existing_summaries + 1:02d}.html"
                    else:
                        filename = f"week_{week_number:02d}_{module_type}.html"
                    
                    # Generate page title
                    title = f"Module {week_number}: {sub_module['title']}"
                    
                    # Generate content based on type
                    if module_type == 'overview':
                        content = self.generate_overview_html(sub_module, week_number)
                    elif module_type == 'concept_summary':
                        content = self.generate_concept_summary_html(sub_module, week_number)
                    elif module_type == 'key_concepts':
                        content = self.generate_key_concepts_html(sub_module, week_number)
                    elif module_type == 'visual_content':
                        content = self.generate_visual_content_html(sub_module, week_number)
                    elif module_type == 'application_examples':
                        content = self.generate_application_examples_html(sub_module, week_number)
                    elif module_type == 'real_world':
                        content = self.generate_real_world_html(sub_module, week_number)
                    elif module_type == 'study_questions':
                        content = self.generate_study_questions_html(sub_module, week_number)
                    else:
                        # Default content for unknown types
                        content = self.format_content_paragraphs(sub_module.get('content', ''))
                    
                    # Generate complete HTML
                    html_content = self.generate_html_template(title, content, module_type)
                    
                    # Write HTML file
                    file_path = output_path / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    generated_files.append(filename)
                    self.logger.info(f"Generated: {filename}")
            
            # Validate output
            self.validate_html_output(output_path, generated_files)
            
            result = {
                "status": "success",
                "files_generated": len(generated_files),
                "output_directory": str(output_path),
                "generated_files": generated_files,
                "total_weeks": len(course_data['weeks']),
                "generation_time": "2025-08-05"
            }
            
            self.logger.info(f"Successfully generated {len(generated_files)} HTML files")
            return result
            
        except Exception as e:
            self.cleanup_temps()
            raise SystemExit(f"HTML GENERATION FAILED: {e}")
        
        finally:
            if self.execution_lock and self.execution_lock.exists():
                self.execution_lock.unlink()
    
    def validate_html_output(self, output_path: Path, generated_files: List[str]):
        """
        Validate generated HTML files meet requirements.
        
        Args:
            output_path (Path): Output directory path
            generated_files (list): List of generated filenames
            
        Raises:
            SystemExit: If validation fails
        """
        self.logger.info("Validating HTML output")
        
        # Check all files exist
        for filename in generated_files:
            file_path = output_path / filename
            if not file_path.exists():
                raise SystemExit(f"VALIDATION ERROR: Generated file missing: {filename}")
        
        # Basic HTML validation
        for filename in generated_files[:3]:  # Sample validation
            file_path = output_path / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for required HTML structure
            required_elements = ['<!DOCTYPE html>', '<html', '<head>', '<body>', '</html>']
            for element in required_elements:
                if element not in content:
                    raise SystemExit(f"VALIDATION ERROR: Missing required HTML element '{element}' in {filename}")
            
            # Check for Bootstrap framework
            if 'bootstrap' not in content.lower():
                raise SystemExit(f"VALIDATION ERROR: Bootstrap framework missing in {filename}")
        
        self.logger.info("HTML output validation passed")
    
    def cleanup_temps(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink()
            except FileNotFoundError:
                pass

def main():
    """Command line interface for HTML generator."""
    parser = argparse.ArgumentParser(description='Generate HTML files from structured course content')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--config', help='Configuration file path (optional)')
    
    args = parser.parse_args()
    
    html_generator = HTMLGenerator(args.config)
    result = html_generator.generate_html_files(args.input, args.output)
    
    print(f"Successfully generated {result['files_generated']} HTML files in {result['output_directory']}")

if __name__ == "__main__":
    main()