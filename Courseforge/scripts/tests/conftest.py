"""
Courseforge Test Fixtures
Shared pytest fixtures for all test modules
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Fixture directories
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
SAMPLE_HTML_DIR = FIXTURES_DIR / 'sample_html'
SAMPLE_IMSCC_DIR = FIXTURES_DIR / 'sample_imscc'


# =============================================================================
# HTML FIXTURE PATHS
# =============================================================================

@pytest.fixture
def accessible_html_path():
    """Path to fully WCAG AA compliant HTML file"""
    return SAMPLE_HTML_DIR / 'accessible.html'


@pytest.fixture
def missing_alt_html_path():
    """Path to HTML with missing alt attributes on images"""
    return SAMPLE_HTML_DIR / 'missing_alt.html'


@pytest.fixture
def broken_headings_html_path():
    """Path to HTML with improper heading hierarchy"""
    return SAMPLE_HTML_DIR / 'broken_headings.html'


@pytest.fixture
def forms_no_labels_html_path():
    """Path to HTML with forms lacking proper labels"""
    return SAMPLE_HTML_DIR / 'forms_no_labels.html'


# =============================================================================
# HTML CONTENT FIXTURES (in-memory)
# =============================================================================

@pytest.fixture
def accessible_html_content():
    """Returns WCAG AA compliant HTML content"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessible Test Page</title>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <header role="banner">
        <nav aria-label="Main navigation">
            <ul>
                <li><a href="index.html">Home</a></li>
                <li><a href="about.html">About Us</a></li>
            </ul>
        </nav>
    </header>
    <main id="main-content" role="main">
        <h1>Welcome to the Course</h1>
        <section aria-labelledby="intro-heading">
            <h2 id="intro-heading">Introduction</h2>
            <p>This is an introduction paragraph with sufficient content.</p>
            <img src="diagram.png" alt="Network topology diagram showing three connected servers">
            <h3>Key Concepts</h3>
            <ul>
                <li>First concept explanation</li>
                <li>Second concept explanation</li>
            </ul>
        </section>
        <section aria-labelledby="form-heading">
            <h2 id="form-heading">Contact Form</h2>
            <form action="/submit" method="post">
                <div>
                    <label for="name">Full Name:</label>
                    <input type="text" id="name" name="name" required aria-required="true">
                </div>
                <div>
                    <label for="email">Email Address:</label>
                    <input type="email" id="email" name="email" required aria-required="true">
                </div>
                <button type="submit">Submit Form</button>
            </form>
        </section>
        <section aria-labelledby="table-heading">
            <h2 id="table-heading">Data Table</h2>
            <table>
                <caption>Quarterly Sales Data</caption>
                <thead>
                    <tr>
                        <th scope="col">Quarter</th>
                        <th scope="col">Revenue</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th scope="row">Q1</th>
                        <td>$10,000</td>
                    </tr>
                    <tr>
                        <th scope="row">Q2</th>
                        <td>$12,000</td>
                    </tr>
                </tbody>
            </table>
        </section>
    </main>
    <footer role="contentinfo">
        <p>&copy; 2025 Course Provider</p>
    </footer>
</body>
</html>'''


@pytest.fixture
def missing_alt_html_content():
    """Returns HTML with images missing alt attributes"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Missing Alt Text Test</title>
</head>
<body>
    <main>
        <h1>Page with Missing Alt Text</h1>
        <img src="image1.png">
        <img src="image2.jpg" alt="">
        <img src="image3.gif">
        <p>Some content here.</p>
    </main>
</body>
</html>'''


@pytest.fixture
def broken_headings_html_content():
    """Returns HTML with improper heading hierarchy (skips levels)"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Broken Headings Test</title>
</head>
<body>
    <main>
        <h1>Main Title</h1>
        <h3>Skipped to H3</h3>
        <p>Content under H3</p>
        <h5>Skipped to H5</h5>
        <p>Content under H5</p>
        <h2>Back to H2</h2>
        <h4>Skipped to H4</h4>
    </main>
</body>
</html>'''


@pytest.fixture
def forms_no_labels_html_content():
    """Returns HTML with form inputs lacking proper labels"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Forms Without Labels Test</title>
</head>
<body>
    <main>
        <h1>Form Without Labels</h1>
        <form action="/submit" method="post">
            <input type="text" name="username" placeholder="Username">
            <input type="password" name="password" placeholder="Password">
            <select name="country">
                <option value="">Select Country</option>
                <option value="us">United States</option>
            </select>
            <textarea name="comments" placeholder="Comments"></textarea>
            <input type="checkbox" name="agree" value="yes"> I agree
            <button type="submit">Submit</button>
        </form>
    </main>
</body>
</html>'''


# =============================================================================
# IMSCC FIXTURES
# =============================================================================

@pytest.fixture
def minimal_manifest_content():
    """Returns minimal valid IMSCC manifest XML"""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="test_course_manifest"
          xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
          xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest">
    <metadata>
        <schema>IMS Common Cartridge</schema>
        <schemaversion>1.2.0</schemaversion>
        <lomimscc:lom>
            <lomimscc:general>
                <lomimscc:title>
                    <lomimscc:string language="en">Test Course</lomimscc:string>
                </lomimscc:title>
            </lomimscc:general>
        </lomimscc:lom>
    </metadata>
    <organizations>
        <organization identifier="org_1" structure="rooted-hierarchy">
            <item identifier="root">
                <item identifier="week1_item" identifierref="week1_resource">
                    <title>Week 1: Introduction</title>
                </item>
            </item>
        </organization>
    </organizations>
    <resources>
        <resource identifier="week1_resource" type="webcontent" href="week_01/introduction.html">
            <file href="week_01/introduction.html"/>
        </resource>
    </resources>
</manifest>'''


@pytest.fixture
def qti_assessment_content():
    """Returns minimal valid QTI 1.2 assessment XML"""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
    <assessment ident="test_quiz" title="Sample Quiz">
        <qtimetadata>
            <qtimetadatafield>
                <fieldlabel>cc_profile</fieldlabel>
                <fieldentry>cc.exam.v0p1</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
                <fieldlabel>qmd_assessmenttype</fieldlabel>
                <fieldentry>Examination</fieldentry>
            </qtimetadatafield>
        </qtimetadata>
        <section ident="section_1">
            <item ident="question_1" title="Sample Question">
                <itemmetadata>
                    <qtimetadata>
                        <qtimetadatafield>
                            <fieldlabel>question_type</fieldlabel>
                            <fieldentry>multiple_choice_question</fieldentry>
                        </qtimetadatafield>
                    </qtimetadata>
                </itemmetadata>
                <presentation>
                    <material>
                        <mattext texttype="text/html">What is 2 + 2?</mattext>
                    </material>
                    <response_lid ident="response_1" rcardinality="Single">
                        <render_choice>
                            <response_label ident="choice_1">
                                <material><mattext>3</mattext></material>
                            </response_label>
                            <response_label ident="choice_2">
                                <material><mattext>4</mattext></material>
                            </response_label>
                        </render_choice>
                    </response_lid>
                </presentation>
                <resprocessing>
                    <respcondition>
                        <conditionvar>
                            <varequal respident="response_1">choice_2</varequal>
                        </conditionvar>
                        <setvar action="Set">100</setvar>
                    </respcondition>
                </resprocessing>
            </item>
        </section>
    </assessment>
</questestinterop>'''


# =============================================================================
# TEMPORARY DIRECTORY FIXTURES
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates a temporary output directory that is cleaned up after test"""
    output_dir = tmp_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def temp_course_dir(tmp_path, accessible_html_content):
    """Creates a mock course directory structure"""
    course_dir = tmp_path / 'test_course'
    course_dir.mkdir(parents=True, exist_ok=True)

    # Create week directories
    for week in range(1, 4):
        week_dir = course_dir / f'week_{week:02d}'
        week_dir.mkdir(parents=True, exist_ok=True)

        # Create module files
        for module in range(1, 4):
            module_file = week_dir / f'module_{module:02d}.html'
            module_file.write_text(accessible_html_content)

    yield course_dir


@pytest.fixture
def temp_imscc_package(tmp_path, minimal_manifest_content, accessible_html_content):
    """Creates a minimal IMSCC package for testing extraction"""
    import zipfile

    package_dir = tmp_path / 'package_content'
    package_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest_path = package_dir / 'imsmanifest.xml'
    manifest_path.write_text(minimal_manifest_content)

    # Create content directory
    week_dir = package_dir / 'week_01'
    week_dir.mkdir(parents=True, exist_ok=True)

    # Create HTML content
    html_path = week_dir / 'introduction.html'
    html_path.write_text(accessible_html_content)

    # Create IMSCC zip file
    imscc_path = tmp_path / 'test_course.imscc'
    with zipfile.ZipFile(imscc_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)

    yield imscc_path


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def write_temp_html(tmp_path):
    """Factory fixture for writing temporary HTML files"""
    def _write_html(content, filename='test.html'):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _write_html


@pytest.fixture
def courseforge_path():
    """Returns the path to the Courseforge project root"""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def scripts_path():
    """Returns the path to the scripts directory"""
    return Path(__file__).parent.parent
