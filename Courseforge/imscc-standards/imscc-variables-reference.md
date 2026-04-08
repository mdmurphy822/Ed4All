# IMSCC Variables and Formatting Reference

## Complete Variable Reference for IMS Common Cartridge Implementation

This document provides a comprehensive reference of all variables, attributes, and formatting options available in IMSCC packages across versions 1.1, 1.2, and 1.3.

**UPDATED**: Now includes actual Brightspace D2L export formatting based on real exported package analysis.

## Manifest Root Element Variables

### Required Attributes (Brightspace D2L Format)
```xml
<manifest identifier="i53063987-612a-477d-9c3a-86d2d8471636" 
          xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1" 
          xmlns:lomr="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource" 
          xmlns:lomm="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest" 
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
          xsi:schemaLocation="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd 
                              http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1 http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd 
                              http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd">
```

**Variable Details**:
- `identifier`: Unique GUID-format identifier (e.g., "i53063987-612a-477d-9c3a-86d2d8471636")
- `xmlns`: Primary namespace URI (IMSCC 1.3 format)
- `xmlns:lomr`: LOM Resource namespace (note: "lomr" not "lom")
- `xmlns:lomm`: LOM Manifest namespace (note: "lomm" not "lomimscc")
- `xmlns:xsi`: XML Schema Instance namespace
- `xsi:schemaLocation`: Full schema location mapping (required for Brightspace)

### Version-Specific Namespaces

#### IMSCC 1.1
```xml
xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource"
xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest"
```

#### IMSCC 1.2
```xml
xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest"
```

#### IMSCC 1.3
```xml
xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
```

## Metadata Section Variables

### Required Schema Variables (Brightspace D2L Format)
```xml
<metadata>
  <schema>IMS Common Cartridge</schema>
  <schemaversion>1.3.0</schemaversion>
  <lomm:lom>
    <lomm:general>
      <lomm:title>
        <lomm:string language="en-US">Course Title</lomm:string>
      </lomm:title>
    </lomm:general>
  </lomm:lom>
</metadata>
```

**Schema Version Values**:
- `1.1.0` for IMSCC 1.1
- `1.2.0` for IMSCC 1.2  
- `1.3.0` for IMSCC 1.3 (Brightspace default)

**Key Brightspace Differences**:
- Uses `lomm:lom` instead of `lomimscc:lom`
- Language attribute format: `"en-US"` instead of `"en"`
- Simplified metadata structure compared to full Dublin Core

## Brightspace-Specific Resource Types (From Actual Export)

### Assignment Resources
```xml
<resource identifier="iabe8c68b-1bdc-42c4-a1de-1f67b84ed567_R" type="assignment_xmlv1p0" href="assignment/folder/assignment.xml">
  <file href="assignment/i913b681b-261a-4d7d-aa63-32d7c783e726/assignment_73a72635-e939-4e11-94f6-1fb9b875d646.xml"/>
</resource>
```
**Key Properties**:
- `type`: `"assignment_xmlv1p0"` (not D2L 2.0 format)
- Namespace: `"http://www.imsglobal.org/xsd/imscc_extensions/assignment"`
- Schema Location: `"http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd"`

### Discussion Resources  
```xml
<resource identifier="i548f7842-5f9e-4770-a830-b0b5e22cdc52_R" type="imsdt_xmlv1p3" href="discussion/folder/discussion.xml">
  <file href="discussion/ia692bed5-443d-4485-a3dd-277a831d6e3e/discussion_904514a2-16ee-411e-bd6f-2d63361b316c.xml"/>
</resource>
```
**Key Properties**:
- `type`: `"imsdt_xmlv1p3"` (IMSCC 1.3 discussion format)
- Namespace: `"http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"`
- Schema Location: `"http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd"`

### Quiz/Assessment Resources
```xml
<resource identifier="i62bca3a6-1d6d-47e6-8a85-9b08fd4d71a2_R" type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment" href="quiz/folder/qti.xml">
  <file href="quiz/i7f9e1e58-7228-4768-a27d-a89c215d41dc/qti_e0766f3a-fbe9-4cf4-942e-42847f225b13.xml"/>
</resource>
```
**Key Properties**:
- `type`: `"imsqti_xmlv1p2/imscc_xmlv1p3/assessment"` (mixed QTI/IMSCC format)
- Namespace: `"http://www.imsglobal.org/xsd/ims_qtiasiv1p2"`
- Schema Location: `"http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd"`

### Content File Structure (From Export)
```
├── content/
│   ├── i22116317-3514-40e3-a951-85d2c3420291/
│   │   └── objective_1_1_osi_model.html
│   ├── ic3da1bc7-e9e7-44e4-b617-fd6b83cfd3f5/
│   │   └── objective_1_2_networking_appliances.html
│   └── [additional GUID-named folders]
├── assignment/
│   └── i913b681b-261a-4d7d-aa63-32d7c783e726/
│       └── assignment_73a72635-e939-4e11-94f6-1fb9b875d646.xml
├── discussion/
│   └── ia692bed5-443d-4485-a3dd-277a831d6e3e/
│       └── discussion_904514a2-16ee-411e-bd6f-2d63361b316c.xml
└── quiz/
    └── i7f9e1e58-7228-4768-a27d-a89c215d41dc/
        └── qti_e0766f3a-fbe9-4cf4-942e-42847f225b13.xml
```

**Content Organization Patterns**:
- All content files in GUID-named subfolders (prefix "i" + 32-character hex)
- Assignment files use `assignment_` prefix in XML filename
- Discussion files use `discussion_` prefix in XML filename  
- Quiz files use `qti_` prefix in XML filename
- All folders follow two-level directory structure

## CRITICAL: Project Update Summary

**IMPORTANT DISCOVERY**: Our previous D2L 2.0 format documentation was incorrect. Brightspace actually uses:

### What We Previously Documented (INCORRECT):
- D2L 2.0 namespace: `"http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"`
- Complex D2L assignment/discussion schemas with gradebook metadata
- D2L-specific QTI metadata fields

### What Brightspace Actually Uses (CORRECT):
- Standard IMSCC 1.3 extensions: `"http://www.imsglobal.org/xsd/imscc_extensions/assignment"`
- Simplified assignment XML with basic gradable points
- Standard QTI 1.2 with cc_profile metadata (not D2L-specific)
- Discussion topics as `<topic>` elements (not `<discussion>`)

### Updated Resource Type Mappings:
```xml
<!-- Assignment Resources -->
<resource type="assignment_xmlv1p0" href="assignment.xml"/>

<!-- Discussion Resources -->
<resource type="imsdt_xmlv1p3" href="discussion.xml"/>

<!-- Quiz Resources -->
<resource type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment" href="quiz.xml"/>
```

### Agent Impact:
- **brightspace-packager**: Must use new resource types and namespaces
- **assessment generators**: Must output simplified XML format (not D2L 2.0)
- **manifest generators**: Must use IMSCC 1.3 with lomr/lomm namespaces
- **validation scripts**: Must check against actual Brightspace import format

**ACTION REQUIRED**: All generation scripts updated to use actual Brightspace export format.

### Dublin Core Metadata Variables (15 Elements)

```xml
<lomimscc:lom>
  <lomimscc:general>
    <lomimscc:identifier>
      <lomimscc:catalog>URI</lomimscc:catalog>
      <lomimscc:entry>unique_course_id</lomimscc:entry>
    </lomimscc:identifier>
    <lomimscc:title>
      <lomimscc:string language="en">Course Title</lomimscc:string>
    </lomimscc:title>
    <lomimscc:language>en</lomimscc:language>
    <lomimscc:description>
      <lomimscc:string language="en">Course Description</lomimscc:string>
    </lomimscc:description>
    <lomimscc:keyword>
      <lomimscc:string language="en">keyword1, keyword2</lomimscc:string>
    </lomimscc:keyword>
  </lomimscc:general>
  <lomimscc:lifeCycle>
    <lomimscc:version>
      <lomimscc:string language="en">1.0</lomimscc:string>
    </lomimscc:version>
    <lomimscc:status>
      <lomimscc:source>LOMv1.0</lomimscc:source>
      <lomimscc:value>final</lomimscc:value>
    </lomimscc:status>
    <lomimscc:contribute>
      <lomimscc:role>
        <lomimscc:source>LOMv1.0</lomimscc:source>
        <lomimscc:value>author</lomimscc:value>
      </lomimscc:role>
      <lomimscc:entity>BEGIN:VCARD
FN:Author Name
EMAIL:author@example.com
END:VCARD</lomimscc:entity>
      <lomimscc:date>
        <lomimscc:dateTime>2024-01-15</lomimscc:dateTime>
      </lomimscc:date>
    </lomimscc:contribute>
  </lomimscc:lifeCycle>
  <lomimscc:technical>
    <lomimscc:format>text/html</lomimscc:format>
    <lomimscc:location>index.html</lomimscc:location>
  </lomimscc:technical>
  <lomimscc:educational>
    <lomimscc:intendedEndUserRole>
      <lomimscc:source>LOMv1.0</lomimscc:source>
      <lomimscc:value>learner</lomimscc:value>
    </lomimscc:intendedEndUserRole>
    <lomimscc:context>
      <lomimscc:source>LOMv1.0</lomimscc:source>
      <lomimscc:value>higher education</lomimscc:value>
    </lomimscc:context>
    <lomimscc:typicalAgeRange>
      <lomimscc:string language="en">18-65</lomimscc:string>
    </lomimscc:typicalAgeRange>
    <lomimscc:difficulty>
      <lomimscc:source>LOMv1.0</lomimscc:source>
      <lomimscc:value>medium</lomimscc:value>
    </lomimscc:difficulty>
    <lomimscc:typicalLearningTime>
      <lomimscc:duration>PT15W</lomimscc:duration>
    </lomimscc:typicalLearningTime>
  </lomimscc:educational>
  <lomimscc:rights>
    <lomimscc:cost>
      <lomimscc:source>LOMv1.0</lomimscc:source>
      <lomimscc:value>no</lomimscc:value>
    </lomimscc:cost>
    <lomimscc:copyrightAndOtherRestrictions>
      <lomimscc:source>LOMv1.0</lomimscc:source>
      <lomimscc:value>yes</lomimscc:value>
    </lomimscc:copyrightAndOtherRestrictions>
    <lomimscc:description>
      <lomimscc:string language="en">Copyright notice</lomimscc:string>
    </lomimscc:description>
  </lomimscc:rights>
</lomimscc:lom>
```

### Curriculum Standards Metadata (IMSCC 1.2+)
```xml
<curriculumStandardsMetadataSet>
  <curriculumStandardsMetadata>
    <setOfGUIDs>
      <labelledGUID>
        <GUID>standard_guid_123</GUID>
        <label>Common Core Math Standard</label>
      </labelledGUID>
    </setOfGUIDs>
  </curriculumStandardsMetadata>
</curriculumStandardsMetadataSet>
```

## Organization Section Variables

### Organization Element Attributes
```xml
<organizations default="ORG_IDENTIFIER">
  <organization identifier="ORG_IDENTIFIER" structure="rooted-hierarchy">
    <title>Organization Title</title>
    <!-- Items go here -->
  </organization>
</organizations>
```

**Organization Variables**:
- `default`: Identifier of the default organization
- `identifier`: Unique identifier for this organization
- `structure`: Must be "rooted-hierarchy"

### Item Element Variables
```xml
<item identifier="ITEM_ID" 
      identifierref="RESOURCE_ID" 
      isvisible="true"
      parameters="param1=value1&param2=value2">
  <title>Item Title</title>
  <adlcp:timeLimitAction>continue</adlcp:timeLimitAction>
  <adlcp:dataFromLMS>course_id={course_id}</adlcp:dataFromLMS>
  <!-- Nested items -->
</item>
```

**Item Attributes**:
- `identifier`: Unique identifier for the item
- `identifierref`: References a resource in the resources section
- `isvisible`: Boolean (true/false) - controls visibility
- `parameters`: URL-encoded parameter string

**Item Sub-elements**:
- `<title>`: Display title for the item
- `<adlcp:timeLimitAction>`: Values include "exit", "continue", "message"
- `<adlcp:dataFromLMS>`: Data passed from LMS to content

## Resources Section Variables

### Resource Element Attributes
```xml
<resource identifier="RESOURCE_ID" 
          type="RESOURCE_TYPE" 
          href="primary_file.html"
          xml:base="subfolder/"
          adlcp:scormType="sco">
  <metadata>
    <!-- Resource-specific metadata -->
  </metadata>
  <file href="file1.html"/>
  <file href="file2.css"/>
  <dependency identifierref="DEPENDENT_RESOURCE_ID"/>
</resource>
```

**Resource Attributes**:
- `identifier`: Unique identifier for the resource
- `type`: Resource type (see Resource Types section)
- `href`: Primary file for the resource
- `xml:base`: Base path for relative file references
- `adlcp:scormType`: SCORM compatibility ("sco" or "asset")

### Resource Types (Complete List)

#### Standard Resource Types
- `webcontent`: Web-based learning content
- `associatedcontent/imscc_xmlv1p1/learning-application-resource`: Associated content
- `imsqti_xmlv1p2/imscc_xmlv1p1/assessment`: QTI 1.2 Assessment
- `imsqti_xmlv1p2/imscc_xmlv1p1/question-bank`: QTI 1.2 Question Bank
- `imsdt_xmlv1p1`: Discussion Topic
- `imswl_xmlv1p1`: Web Link
- `imsbasiclti_xmlv1p0`: Basic LTI Link

#### Brightspace D2L Resource Types
- `imsccv1p1/d2l_2p0/assignment`: D2L Assignment
- `imsccv1p1/d2l_2p0/discussion`: D2L Discussion
- `imsccv1p1/d2l_2p0/quiz`: D2L Quiz
- `imsccv1p1/d2l_2p0/survey`: D2L Survey
- `imsccv1p1/d2l_2p0/dropbox`: D2L Dropbox

#### Canvas-Specific Resource Types
- `imsccv1p1/canvas_xmlv1p0/assignment`: Canvas Assignment
- `imsccv1p1/canvas_xmlv1p0/discussion_topic`: Canvas Discussion
- `imsccv1p1/canvas_xmlv1p0/quiz`: Canvas Quiz
- `imsccv1p1/canvas_xmlv1p0/module_meta`: Canvas Module Metadata

### File Element Variables
```xml
<file href="relative/path/to/file.html" 
      size="12345" 
      checksum="md5hash"
      checksumtype="MD5"/>
```

**File Attributes**:
- `href`: Relative path to the file
- `size`: File size in bytes (optional)
- `checksum`: File checksum (optional)
- `checksumtype`: Checksum algorithm (MD5, SHA1, etc.)

### Dependency Variables
```xml
<dependency identifierref="DEPENDENT_RESOURCE_ID"/>
```

**Dependency Attributes**:
- `identifierref`: References another resource this resource depends on

## Web Content Variables

### IntendedUse Attribute Values
```xml
<resource type="webcontent" intendeduse="VALUE">
```

**IntendedUse Values**:
- `lessonplan`: Lesson plan content
- `syllabus`: Course syllabus
- `assignment`: Assignment instructions (IMSCC 1.2+)
- `unspecified`: General content

### Educational Metadata for Web Content
```xml
<lom:lom>
  <lom:educational>
    <lom:intendedEndUserRole>
      <lom:source>LOMv1.0</lom:source>
      <lom:value>learner</lom:value>
    </lom:intendedEndUserRole>
    <lom:context>
      <lom:source>LOMv1.0</lom:source>
      <lom:value>higher education</lom:value>
    </lom:context>
    <lom:difficulty>
      <lom:source>LOMv1.0</lom:source>
      <lom:value>medium</lom:value>
    </lom:difficulty>
  </lom:educational>
</lom:lom>
```

**Educational Role Values**:
- `teacher`: Teacher/instructor
- `author`: Content author
- `learner`: Student/learner
- `manager`: Course manager

**Context Values**:
- `school`: K-12 education
- `higher education`: College/university
- `training`: Professional training
- `other`: Other contexts

**Difficulty Values**:
- `very easy`: Very easy
- `easy`: Easy
- `medium`: Medium
- `difficult`: Difficult
- `very difficult`: Very difficult

## Assessment Variables (QTI)

### QTI Assessment Resource
```xml
<resource identifier="ASSESSMENT_ID" 
          type="imsqti_xmlv1p2/imscc_xmlv1p1/assessment" 
          href="assessment.xml">
  <dependency identifierref="QTI_ASI_BASE"/>
  <file href="assessment.xml"/>
</resource>
```

### QTI Assessment XML Variables
```xml
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
  <assessment ident="ASSESSMENT_ID" title="Assessment Title">
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_maxattempts</fieldlabel>
        <fieldentry>3</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>cc_timelimit</fieldlabel>
        <fieldentry>60</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>points_possible</fieldlabel>
        <fieldentry>100</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="SECTION_ID">
      <item ident="ITEM_ID" title="Question Title">
        <!-- Question content -->
      </item>
    </section>
  </assessment>
</questestinterop>
```

### QTI Metadata Fields
- `cc_maxattempts`: Maximum attempts allowed
- `cc_timelimit`: Time limit in minutes
- `points_possible`: Total points for assessment
- `bb_question_type`: Question type (for Blackboard compatibility)
- `assessment_type`: Type of assessment (survey, graded_survey, practice_quiz, etc.)

## Brightspace D2L Specific Variables

### D2L Assignment Variables
```xml
<assignment xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0">
  <gradebook_item>
    <points_possible>100.0</points_possible>
    <weight>10.0</weight>
    <due_date>2024-12-31T23:59:59Z</due_date>
    <is_bonus>false</is_bonus>
    <exclude_from_final_grade>false</exclude_from_final_grade>
  </gradebook_item>
  <dropbox>
    <dropbox_type>Individual</dropbox_type>
    <submissions_allowed>unlimited</submissions_allowed>
    <categorized_submissions>false</categorized_submissions>
    <anonymous_marking>false</anonymous_marking>
  </dropbox>
  <submission_view>
    <allow_submissions_from_date>2024-01-01T00:00:00Z</allow_submissions_from_date>
    <allow_submissions_until_date>2024-12-31T23:59:59Z</allow_submissions_until_date>
  </submission_view>
</assignment>
```

**D2L Assignment Variables**:
- `points_possible`: Maximum points (decimal)
- `weight`: Grade weight percentage
- `due_date`: ISO 8601 datetime
- `is_bonus`: Boolean
- `dropbox_type`: "Individual" or "Group"
- `submissions_allowed`: Number or "unlimited"
- `anonymous_marking`: Boolean

### D2L Discussion Variables
```xml
<discussion xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0">
  <gradebook_item>
    <points_possible>25.0</points_possible>
  </gradebook_item>
  <forum_settings>
    <forum_type>Topic</forum_type>
    <allow_anonymous>false</allow_anonymous>
    <must_post_first>true</must_post_first>
    <auto_lock_date>2024-12-31T23:59:59Z</auto_lock_date>
  </forum_settings>
  <participation_requirements>
    <initial_post_required>true</initial_post_required>
    <replies_required>2</replies_required>
    <reply_to_count>1</reply_to_count>
  </participation_requirements>
  <scoring_settings>
    <score_out_of>25</score_out_of>
    <posts_have_scores>true</posts_have_scores>
  </scoring_settings>
</discussion>
```

**D2L Discussion Variables**:
- `forum_type`: "Topic", "QA", "General"
- `allow_anonymous`: Boolean
- `must_post_first`: Boolean
- `initial_post_required`: Boolean
- `replies_required`: Number
- `score_out_of`: Maximum score

### D2L QTI Extensions
```xml
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
    <fieldentry>quiz_password</fieldentry>
  </qtimetadatafield>
  <qtimetadatafield>
    <fieldlabel>d2l_2p0_can_exceed_max_points</fieldlabel>
    <fieldentry>false</fieldentry>
  </qtimetadatafield>
</qtimetadata>
```

## Discussion Topic Variables

### Discussion XML Structure
```xml
<dt:topic xmlns:dt="http://www.imsglobal.org/xsd/imsdt_v1p0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dt:title>Discussion Title</dt:title>
  <dt:text texttype="text/html">
    <![CDATA[Discussion prompt content]]>
  </dt:text>
  <dt:attachments>
    <dt:attachment href="attachment.pdf"/>
  </dt:attachments>
</dt:topic>
```

**Discussion Variables**:
- `texttype`: "text/html" or "text/plain"
- Content wrapped in CDATA for HTML content

## Basic LTI Variables

### Basic LTI XML Structure
```xml
<cartridge_basiclti_link xmlns="http://www.imsglobal.org/xsd/imslticc_v1p0"
                         xmlns:blti="http://www.imsglobal.org/xsd/imsbasiclti_v1p0"
                         xmlns:lticm="http://www.imsglobal.org/xsd/imslticm_v1p0"
                         xmlns:lticp="http://www.imsglobal.org/xsd/imslticp_v1p0">
  <blti:title>LTI Tool Title</blti:title>
  <blti:description>Tool description</blti:description>
  <blti:icon>tool_icon.png</blti:icon>
  <blti:launch_url>https://tool.example.com/launch</blti:launch_url>
  <blti:secure_launch_url>https://tool.example.com/secure_launch</blti:secure_launch_url>
  <blti:vendor>
    <lticp:code>vendor_code</lticp:code>
    <lticp:name>Vendor Name</lticp:name>
    <lticp:description>Vendor description</lticp:description>
    <lticp:url>https://vendor.example.com</lticp:url>
    <lticp:contact>
      <lticp:email>support@vendor.example.com</lticp:email>
    </lticp:contact>
  </blti:vendor>
  <blti:custom>
    <lticm:property name="custom_param1">value1</lticm:property>
    <lticm:property name="custom_param2">value2</lticm:property>
  </blti:custom>
  <blti:extensions platform="canvas.instructure.com">
    <lticm:property name="privacy_level">public</lticm:property>
    <lticm:property name="course_navigation">
      <lticm:property name="enabled">true</lticm:property>
      <lticm:property name="text">Tool Name</lticm:property>
    </lticm:property>
  </blti:extensions>
</cartridge_basiclti_link>
```

**LTI Variables**:
- `launch_url`: HTTP launch URL
- `secure_launch_url`: HTTPS launch URL
- `privacy_level`: "anonymous", "name_only", "public"
- Custom properties for tool configuration

## Web Links Variables

### Web Link XML Structure
```xml
<webLink xmlns="http://www.imsglobal.org/xsd/imswl_v1p0">
  <title>Link Title</title>
  <url href="https://example.com" target="_blank"/>
  <description>Link description</description>
</webLink>
```

**Web Link Variables**:
- `href`: Target URL
- `target`: "_blank", "_self", "_parent", "_top"

## File Path and Naming Variables

### Supported File Extensions
- `.html`, `.htm`: HTML content
- `.xml`: XML data files
- `.css`: Stylesheets
- `.js`: JavaScript files
- `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`: Images
- `.pdf`: PDF documents
- `.mp4`, `.mp3`, `.wav`: Media files
- `.zip`: Compressed archives (for resources)

### Path Constraints
- All paths must be relative (no leading `/`)
- No parent directory references (`../`)
- Case-sensitive file names
- No spaces in file names (use `_` or `-`)
- UTF-8 encoding for file names
- Maximum path length: 260 characters

## Date and Time Format Variables

### ISO 8601 DateTime Format
```xml
<dateTime>2024-01-15T14:30:00Z</dateTime>
<dateTime>2024-01-15T14:30:00+05:00</dateTime>
```

### Duration Format (ISO 8601)
```xml
<duration>PT15W</duration>    <!-- 15 weeks -->
<duration>PT2H30M</duration>  <!-- 2 hours 30 minutes -->
<duration>P1Y2M3D</duration>  <!-- 1 year, 2 months, 3 days -->
```

## Language Code Variables

### ISO 639-1 Language Codes
- `en`: English
- `es`: Spanish  
- `fr`: French
- `de`: German
- `it`: Italian
- `pt`: Portuguese
- `zh`: Chinese
- `ja`: Japanese
- `ko`: Korean
- `ar`: Arabic

### Language with Region
- `en-US`: English (United States)
- `en-GB`: English (United Kingdom)
- `es-ES`: Spanish (Spain)
- `es-MX`: Spanish (Mexico)
- `fr-CA`: French (Canada)

## MIME Type Variables

### Common MIME Types
- `text/html`: HTML files
- `text/css`: CSS files
- `text/javascript`: JavaScript files
- `text/xml`: XML files
- `text/plain`: Plain text files
- `image/png`: PNG images
- `image/jpeg`: JPEG images
- `image/gif`: GIF images
- `image/svg+xml`: SVG images
- `application/pdf`: PDF documents
- `video/mp4`: MP4 videos
- `audio/mp3`: MP3 audio
- `audio/wav`: WAV audio
- `application/zip`: ZIP archives

## Validation and Constraints

### Identifier Constraints
- Must be unique within scope
- Alphanumeric characters plus `_`, `-`, `.`
- No spaces allowed
- Case-sensitive
- Maximum length: 255 characters

### Title Constraints
- UTF-8 encoded text
- Maximum length: 1000 characters
- HTML entities allowed
- Line breaks preserved

### URL Constraints
- Must be valid HTTP/HTTPS URLs
- Relative URLs allowed for internal resources
- Maximum length: 2000 characters
- URL encoding required for special characters

This comprehensive reference provides all available variables and formatting options for IMSCC implementation across all supported versions and LMS platforms.