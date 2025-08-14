# Field Examples Configuration

This document explains how to configure field examples for the MetaBeeAI pipeline to make it more flexible and reusable for different domains.

## Overview

The pipeline uses an external `field_examples.yaml` file to provide examples for different field types instead of hardcoded examples. This makes the system more flexible and allows users to customize examples for their specific domain without modifying the core code.

## File Structure

The `field_examples.yaml` file contains:

- **Field-specific examples**: Detailed examples for specific fields (e.g., `pesticides_data`)
- **Generic examples**: Fallback examples for different field types
- **Default mappings**: Which generic example to use for each field type

## Configuration Sections

### 1. Field Examples

#### Complex Examples
```yaml
complex_examples:
  pesticides_data:
    description: "Example of pesticide data with exposure methods"
    example:
      - name: "imidacloprid"
        exposure_methods:
          - method: "oral"
            doses: [0.1, 1.0, 10.0]
            dose_units: "ng/bee"
            exposure_duration: [48, 72]
            exposure_units: "hours"
```

#### Complex Array Examples
```yaml
complex_array_examples:
  generic_complex_array:
    description: "Generic example for complex array fields"
    example:
      - item_name: "example_item"
        properties:
          - property1: "value1"
          - property2: "value2"
```

#### List Examples
```yaml
list_examples:
  string_list:
    description: "Example of string list fields"
    example: ["item1", "item2", "item3"]
  
  number_list:
    description: "Example of numeric list fields"
    example: [1.0, 2.5, 5.0]
```

#### Simple Examples
```yaml
simple_examples:
  string:
    description: "Example of string fields"
    example: "example_value"
  
  number:
    description: "Example of numeric fields"
    example: 5.2
```

### 2. Default Examples

```yaml
default_examples:
  complex: "generic_complex"
  complex_array: "generic_complex_array"
  list: "string_list"
  number: "number"
  string: "string"
```

## Customizing for Your Domain

### Step 1: Identify Your Fields
Look at your `schema_config.yaml` to see what fields you're extracting and their types.

### Step 2: Create Domain-Specific Examples
Add examples that match your data structure:

```yaml
complex_examples:
  my_custom_field:
    description: "Example of my domain-specific data"
    example:
      - property1: "value1"
        property2: "value2"
        nested_data:
          subproperty: "nested_value"
```

### Step 3: Update Default Examples
If you want different generic examples for your domain:

```yaml
default_examples:
  complex: "my_domain_complex"
  complex_array: "my_domain_array"
  list: "my_domain_list"
  number: "my_domain_number"
  string: "my_domain_string"
```

## Example Use Cases

### Biology Research
```yaml
complex_examples:
  experimental_groups:
    description: "Example of experimental group data"
    example:
      - group_name: "control"
        sample_size: 10
        treatment: "none"
      - group_name: "treatment"
        sample_size: 10
        treatment: "drug_administration"
```

### Chemistry Research
```yaml
complex_examples:
  chemical_compounds:
    description: "Example of chemical compound data"
    example:
      - compound_name: "ethanol"
        molecular_weight: 46.07
        concentration: [0.1, 0.5, 1.0]
        units: "M"
```

### Social Science Research
```yaml
complex_examples:
  survey_responses:
    description: "Example of survey response data"
    example:
      - question: "Age group"
        responses: ["18-25", "26-35", "36-45"]
        counts: [25, 30, 20]
```

## Benefits

1. **Domain Flexibility**: Easy to adapt for different research areas
2. **Maintainability**: Examples are separate from code logic
3. **Customization**: Users can modify examples without touching core code
4. **Reusability**: Generic examples work across different domains
5. **Documentation**: Examples serve as documentation for expected output formats

## Troubleshooting

### Missing Examples File
If the examples file is not found, the system will use fallback examples built from the field configuration.

### Invalid YAML
Check your YAML syntax if you get parsing errors. The system will fall back to default examples.

### Field Not Found
If a specific field example is not found, the system will use the generic example for that field type.

## Migration from Hardcoded Examples

The old hardcoded examples for `pesticides_data` have been moved to the examples file. The system will automatically use these examples if the file is present, maintaining backward compatibility while providing flexibility for future customization.
