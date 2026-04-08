# Brightspace D2L XML Schema Reference

## Complete D2L XML Schema Implementation Guide

This document provides comprehensive reference for implementing Brightspace D2L-specific XML schemas within IMSCC packages, ensuring proper integration with Brightspace tools and functionality.

## Core D2L Namespace Declaration

### Primary D2L Namespace
```xml
xmlns:d2l="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
```

### Complete Namespace Set for D2L Integration
```xml
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
          xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest"
          xmlns:d2l="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
```

## D2L Assignment Schema (Complete Reference)

### Assignment Resource Declaration
```xml
<resource identifier="assignment_001" type="imsccv1p1/d2l_2p0/assignment" href="assignments/assignment_001.xml">
  <file href="assignments/assignment_001.xml"/>
  <file href="assignments/assignment_001_instructions.html"/>
</resource>
```

### Complete Assignment XML Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <!-- Gradebook Integration -->
  <gradebook_item>
    <points_possible>100.0</points_possible>
    <weight>10.0</weight>
    <due_date>2024-12-31T23:59:59Z</due_date>
    <start_date>2024-01-01T00:00:00Z</start_date>
    <end_date>2024-12-31T23:59:59Z</end_date>
    <is_bonus>false</is_bonus>
    <exclude_from_final_grade>false</exclude_from_final_grade>
    <category_id>0</category_id>
    <auto_grade>false</auto_grade>
  </gradebook_item>
  
  <!-- Dropbox Configuration -->
  <dropbox>
    <dropbox_type>Individual</dropbox_type>
    <submissions_allowed>unlimited</submissions_allowed>
    <group_submissions>false</group_submissions>
    <categorized_submissions>false</categorized_submissions>
    <anonymous_marking>false</anonymous_marking>
    <turnitin_enabled>false</turnitin_enabled>
    <feedback_enabled>true</feedback_enabled>
    <rubric_enabled>false</rubric_enabled>
  </dropbox>
  
  <!-- Submission Settings -->
  <submission_view>
    <allow_submissions_from_date>2024-01-01T00:00:00Z</allow_submissions_from_date>
    <allow_submissions_until_date>2024-12-31T23:59:59Z</allow_submissions_until_date>
    <submission_notification>true</submission_notification>
    <completion_type>auto</completion_type>
  </submission_view>
  
  <!-- File Restrictions -->
  <file_restrictions>
    <max_file_size>10485760</max_file_size> <!-- 10MB in bytes -->
    <allowed_file_types>
      <file_type>.pdf</file_type>
      <file_type>.doc</file_type>
      <file_type>.docx</file_type>
      <file_type>.txt</file_type>
    </allowed_file_types>
    <max_files>5</max_files>
  </file_restrictions>
  
  <!-- Instructions Reference -->
  <instructions href="assignment_001_instructions.html"/>
  
</assignment>
```

### Assignment Variable Reference
- `points_possible`: Decimal value for maximum points
- `weight`: Percentage weight in gradebook (decimal)
- `due_date`: ISO 8601 datetime format
- `dropbox_type`: "Individual" or "Group"
- `submissions_allowed`: Integer or "unlimited"
- `turnitin_enabled`: Boolean for plagiarism detection
- `max_file_size`: File size limit in bytes
- `completion_type`: "auto" or "manual"

## D2L Discussion Schema (Complete Reference)

### Discussion Resource Declaration
```xml
<resource identifier="discussion_001" type="imsccv1p1/d2l_2p0/discussion" href="discussions/discussion_001.xml">
  <file href="discussions/discussion_001.xml"/>
  <file href="discussions/discussion_001_prompt.html"/>
</resource>
```

### Complete Discussion XML Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<discussion xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <!-- Gradebook Integration -->
  <gradebook_item>
    <points_possible>25.0</points_possible>
    <weight>5.0</weight>
    <due_date>2024-12-31T23:59:59Z</due_date>
    <is_bonus>false</is_bonus>
    <category_id>0</category_id>
  </gradebook_item>
  
  <!-- Forum Settings -->
  <forum_settings>
    <forum_type>Topic</forum_type>
    <allow_anonymous>false</allow_anonymous>
    <must_post_first>true</must_post_first>
    <display_pinned_first>true</display_pinned_first>
    <auto_lock_date>2024-12-31T23:59:59Z</auto_lock_date>
    <sorting_order>oldest_first</sorting_order>
    <moderation_enabled>false</moderation_enabled>
  </forum_settings>
  
  <!-- Participation Requirements -->
  <participation_requirements>
    <initial_post_required>true</initial_post_required>
    <replies_required>2</replies_required>
    <reply_to_count>1</reply_to_count>
    <minimum_word_count>150</minimum_word_count>
    <rating_enabled>false</rating_enabled>
  </participation_requirements>
  
  <!-- Scoring Settings -->
  <scoring_settings>
    <score_out_of>25</score_out_of>
    <posts_have_scores>true</posts_have_scores>
    <auto_score>false</auto_score>
    <score_threshold>0</score_threshold>
  </scoring_settings>
  
  <!-- Availability Settings -->
  <availability>
    <start_date>2024-01-01T00:00:00Z</start_date>
    <end_date>2024-12-31T23:59:59Z</end_date>
    <visible_to_students>true</visible_to_students>
  </availability>
  
  <!-- Discussion Prompt Reference -->
  <prompt href="discussion_001_prompt.html"/>
  
</discussion>
```

### Discussion Variable Reference
- `forum_type`: "Topic", "QA", "General"
- `must_post_first`: Boolean requiring initial post before viewing others
- `replies_required`: Integer number of required replies
- `minimum_word_count`: Integer word count requirement
- `sorting_order`: "oldest_first", "newest_first", "most_replies"
- `auto_score`: Boolean for automatic scoring
- `moderation_enabled`: Boolean for post moderation

## D2L Quiz/Assessment Schema (QTI + D2L Extensions)

### Quiz Resource Declaration
```xml
<resource identifier="quiz_001" type="imsqti_xmlv1p2/imscc_xmlv1p1/assessment" href="assessments/quiz_001.xml">
  <dependency identifierref="QTI_ASI_BASE"/>
  <file href="assessments/quiz_001.xml"/>
</resource>
```

### Complete QTI Assessment with D2L Extensions
```xml
<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <assessment ident="quiz_001" title="Week 1 Quiz">
    
    <!-- D2L-Specific Metadata -->
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_grade_item_points_possible</fieldlabel>
        <fieldentry>50</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_attempts_allowed</fieldlabel>
        <fieldentry>3</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_time_limit</fieldlabel>
        <fieldentry>60</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_password</fieldlabel>
        <fieldentry></fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_can_exceed_max_points</fieldlabel>
        <fieldentry>false</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_show_correct_answers</fieldlabel>
        <fieldentry>true</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_randomize_questions</fieldlabel>
        <fieldentry>false</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_availability_start_date</fieldlabel>
        <fieldentry>2024-01-01T00:00:00Z</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_availability_end_date</fieldlabel>
        <fieldentry>2024-12-31T23:59:59Z</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_auto_grade</fieldlabel>
        <fieldentry>true</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>d2l_2p0_weight</fieldlabel>
        <fieldentry>10.0</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    
    <section ident="section_001">
      <!-- Quiz questions go here -->
      <item ident="question_001" title="Question 1">
        <!-- Question content -->
      </item>
    </section>
    
  </assessment>
</questestinterop>
```

### D2L QTI Metadata Fields Reference
- `d2l_2p0_grade_item_points_possible`: Maximum points (integer)
- `d2l_2p0_attempts_allowed`: Number of attempts or "unlimited"
- `d2l_2p0_time_limit`: Time limit in minutes (integer)
- `d2l_2p0_password`: Quiz password (string)
- `d2l_2p0_can_exceed_max_points`: Boolean for extra credit
- `d2l_2p0_show_correct_answers`: Boolean for answer display
- `d2l_2p0_randomize_questions`: Boolean for question randomization
- `d2l_2p0_auto_grade`: Boolean for automatic grading
- `d2l_2p0_weight`: Gradebook weight percentage

## D2L Survey Schema

### Survey Resource Declaration
```xml
<resource identifier="survey_001" type="imsccv1p1/d2l_2p0/survey" href="surveys/survey_001.xml">
  <file href="surveys/survey_001.xml"/>
</resource>
```

### Complete Survey XML Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<survey xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <!-- Survey Settings -->
  <survey_settings>
    <anonymous_responses>true</anonymous_responses>
    <multiple_submissions>false</multiple_submissions>
    <show_results_to_students>false</show_results_to_students>
    <availability_start_date>2024-01-01T00:00:00Z</availability_start_date>
    <availability_end_date>2024-12-31T23:59:59Z</availability_end_date>
  </survey_settings>
  
  <!-- Completion Tracking -->
  <completion_tracking>
    <track_completion>true</track_completion>
    <completion_percentage>0</completion_percentage>
  </completion_tracking>
  
</survey>
```

## D2L Content Package Schema

### Content Package Resource Declaration
```xml
<resource identifier="content_001" type="webcontent" href="content/week_01_overview.html">
  <metadata>
    <d2l:d2l_metadata>
      <d2l:content_type>lesson</d2l:content_type>
      <d2l:completion_tracking>true</d2l:completion_tracking>
      <d2l:visibility>visible</d2l:visibility>
    </d2l:d2l_metadata>
  </metadata>
  <file href="content/week_01_overview.html"/>
  <file href="css/styles.css"/>
</resource>
```

## D2L Gradebook Category Schema

### Gradebook Category Configuration
```xml
<gradebook_category xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0">
  <category_id>1</category_id>
  <category_name>Assignments</category_name>
  <weight>50.0</weight>
  <drop_lowest>0</drop_lowest>
  <drop_highest>0</drop_highest>
  <is_bonus>false</is_bonus>
</gradebook_category>
```

## D2L Learning Objectives Schema

### Learning Objectives Integration
```xml
<learning_objectives xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0">
  <objective id="obj_001">
    <title>Understand Linear Transformations</title>
    <description>Students will be able to identify and apply linear transformations</description>
    <proficiency_scale>
      <level value="4">Exceeds Expectations</level>
      <level value="3">Meets Expectations</level>
      <level value="2">Approaching Expectations</level>
      <level value="1">Below Expectations</level>
    </proficiency_scale>
  </objective>
</learning_objectives>
```

## D2L Rubric Schema

### Rubric Resource Declaration
```xml
<resource identifier="rubric_001" type="imsccv1p1/d2l_2p0/rubric" href="rubrics/rubric_001.xml">
  <file href="rubrics/rubric_001.xml"/>
</resource>
```

### Complete Rubric XML Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<rubric xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <rubric_id>rubric_001</rubric_id>
  <rubric_name>Assignment Rubric</rubric_name>
  <rubric_description>Rubric for weekly assignments</rubric_description>
  
  <scoring_method>points</scoring_method>
  <overall_score_visibility>visible</overall_score_visibility>
  
  <criteria>
    <criterion id="criterion_001">
      <criterion_name>Content Quality</criterion_name>
      <criterion_weight>40</criterion_weight>
      <levels>
        <level id="level_001">
          <level_name>Excellent</level_name>
          <level_points>4</level_points>
          <level_description>Exceptional content quality</level_description>
        </level>
        <level id="level_002">
          <level_name>Good</level_name>
          <level_points>3</level_points>
          <level_description>Good content quality</level_description>
        </level>
        <level id="level_003">
          <level_name>Satisfactory</level_name>
          <level_points>2</level_points>
          <level_description>Adequate content quality</level_description>
        </level>
        <level id="level_004">
          <level_name>Needs Improvement</level_name>
          <level_points>1</level_points>
          <level_description>Below expectations</level_description>
        </level>
      </levels>
    </criterion>
  </criteria>
  
</rubric>
```

## Integration Best Practices

### Gradebook Integration Requirements
1. All graded items MUST include `points_possible`
2. Due dates MUST use ISO 8601 format
3. Category assignments should reference valid category IDs
4. Weight percentages should sum to 100% within categories

### Assessment Functionality Requirements
1. QTI assessments MUST include D2L metadata fields
2. Assignment dropboxes MUST specify dropbox_type
3. Discussion forums MUST include participation_requirements
4. Survey anonymous settings should be explicitly defined

### Content Display Requirements
1. HTML content MUST reference valid file paths
2. Organizational hierarchy MUST preserve educational structure
3. Resource types MUST match actual content format
4. Metadata MUST support Brightspace content tracking

## Validation Checklist

### XML Schema Validation
- [ ] All D2L namespace declarations present
- [ ] Resource type declarations match content format
- [ ] Required metadata fields included for each resource type
- [ ] ISO 8601 datetime format used consistently
- [ ] Boolean values properly formatted (true/false)

### Brightspace Integration Validation
- [ ] Assignment dropboxes create in gradebook
- [ ] Quiz assessments integrate with quiz tool
- [ ] Discussion forums support grading
- [ ] Content modules display properly
- [ ] Rubrics associate with assignments correctly

### Educational Content Validation
- [ ] Module structure matches course outline
- [ ] Comprehensive educational content depth verified
- [ ] Subject rigor appropriate for course level
- [ ] Learning objectives align with content
- [ ] Assessment alignment with instructional goals

This comprehensive D2L XML schema reference ensures proper implementation of Brightspace-specific functionality within IMSCC packages, enabling full integration with Brightspace tools and features.