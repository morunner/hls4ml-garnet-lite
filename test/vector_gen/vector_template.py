template_str = """
#pragma once
#include <string>
#include <cstddef>

struct garnet_config {
    static const unsigned V = {{ garnet_config.V }};
    static const unsigned V_nbits = {{ garnet_config.V_nbits }};
    static const unsigned S = {{ garnet_config.S }};
    static const unsigned N = {{ garnet_config.N }};
    static const unsigned exp_table_size = {{ garnet_config.exp_table_size }};
    static const unsigned exp_table_size_nbits = {{ garnet_config.exp_table_size_nbits }};
    static const unsigned exp_table_indexing_shmt = {{ garnet_config.exp_table_indexing_shmt }};
};
{% for class_name, items in data.items() %}
struct {{ class_name }} {
    std::string name;
{%- for field in get_fields(items[0]) %}
    {%- if is_array(items[0], field) %}
    float* {{ field }};
    size_t {{ field }}_len;
    {%- else %}
    {{ get_cpp_type(items[0], field) }} {{ field }};
    {%- endif %}
{%- endfor %}
};
{%- endfor %}
{% for class_name, items in data.items() %}
    {%- for i, item in enumerate(items) %}
        {%- for field in get_fields(item) %}
            {%- if is_array(item, field) %}
static float {{ class_name }}_data_{{ i }}_{{ field }}[] = {{ cpp(get_attr(item, field)) }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
{%- endfor %}
{% for class_name, items in data.items() %}
static const int {{ class_name }}s_length = {{ nvectors(class_name) }};
static {{ class_name }} {{ class_name }}s[] = {
    {%- for i, item in enumerate(items) %}
    {
        "{{ item.name }}",
        {%- for field in get_fields(item) %}
            {%- if is_array(item, field) %}
        {{ class_name }}_data_{{ i }}_{{ field }},
        {{ length(get_attr(item, field)) }},
            {%- else %}
        {{ get_attr(item, field) }},
            {%- endif %}
        {%- endfor %}
    },
    {%- endfor %}
};
{%- endfor %}
"""
