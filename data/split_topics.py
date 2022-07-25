import os
import xml.etree.cElementTree as et
from dicttoxml import dicttoxml
topics_head = ['topic', 'query', 'evidence', 'description', 'narrative', 'stance']
xml_root = et.parse("./data/misinfo-2022-topics.xml")
rows = xml_root.findall('topic')
xml_data = [[dict(number=int(row.find('number').text), query=row.find('question').text)] for row in rows]
for x in xml_data:
    stuff = dicttoxml(x, custom_root='topics', attr_type=False, item_func=lambda _: 'topic')
    with open(f"data/2022_topics_dir/{x[0]['number']}.xml", 'wb') as f:
        f.write(stuff)
