from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base', cache_dir="D:\huggingface_cache")
# model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base', cache_dir="D:\huggingface_cache")

tokenizer.to_device("gpu")

text = """package j\n\nimport (\n\t\"encoding/json\"\n)\n\n\ntype FooJSON struct {\n\tTmp       *Temp `json:\"tmp,omitempty\"`\n\tBar       `json:\",omitempty\"`\n\t*Buzz     `json:\",omitempty\"`\n\tHogeJSON  `json:\",omitempty\"`\n\t*FugaJSON `json:\",omitempty\"`\n}\n\n\ntype FooJSONList []*FooJSON\n\n\ntype FooPropertyEncoder func(src *Foo, dest *FooJSON) error\n\n\ntype FooPropertyDecoder func(src *FooJSON, dest *Foo) error\n\n\ntype FooPropertyInfo struct {\n\tfieldName string\n\tjsonName  string\n\tEncoder   FooPropertyEncoder\n\tDecoder   FooPropertyDecoder\n}\n\n\nfunc (info *FooPropertyInfo) FieldName() string {\n\treturn info.fieldName\n}\n\n\nfunc (info *FooPropertyInfo) JSONName() string {\n\treturn info.jsonName\n}\n\n\ntype FooJSONBuilder struct {\n\t_properties        map[string]*FooPropertyInfo\n\t_jsonPropertyMap   map[string]*FooPropertyInfo\n\t_structPropertyMap map[string]*FooPropertyInfo\n\tTmp                *FooPropertyInfo\n\tBar                *FooPropertyInfo\n\tBuzz               *FooPropertyInfo\n\tHoge               *FooPropertyInfo\n\tFuga               *FooPropertyInfo\n}\n\n\nfunc NewFooJSONBuilder() *FooJSONBuilder {\n\tjb := &FooJSONBuilder{\n\t\t_properties:        map[string]*FooPropertyInfo{},\n\t\t_jsonPropertyMap:   map[string]*FooPropertyInfo{},\n\t\t_structPropertyMap: map[string]*FooPropertyInfo{},\n\t\tTmp: &FooPropertyInfo{\n\t\t\tfieldName: \"Tmp\",\n\t\t\tjsonName:  \"tmp\",\n\t\t\tEncoder: func(src *Foo, dest *FooJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Tmp = src.Tmp\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *FooJSON, dest *Foo) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Tmp = src.Tmp\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t\tBar: &FooPropertyInfo{\n\t\t\tfieldName: \"Bar\",\n\t\t\tjsonName:  \"\",\n\t\t\tEncoder: func(src *Foo, dest *FooJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Bar = src.Bar\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *FooJSON, dest *Foo) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Bar = src.Bar\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t\tBuzz: &FooPropertyInfo{\n\t\t\tfieldName: \"Buzz\",\n\t\t\tjsonName:  \"\",\n\t\t\tEncoder: func(src *Foo, dest *FooJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Buzz = src.Buzz\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *FooJSON, dest *Foo) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Buzz = src.Buzz\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t\tHoge: &FooPropertyInfo{\n\t\t\tfieldName: \"Hoge\",\n\t\t\tjsonName:  \"\",\n\t\t\tEncoder: func(src *Foo, dest *FooJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\td, err := NewHogeJSONBuilder().AddAll().Convert(&src.Hoge)\n\t\t\t\tif err != nil {\n\t\t\t\t\treturn err\n\t\t\t\t}\n\t\t\t\tdest.HogeJSON = *d\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *FooJSON, dest *Foo) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\td, err := src.HogeJSON.Convert()\n\t\t\t\tif err != nil {\n\t\t\t\t\treturn err\n\t\t\t\t}\n\t\t\t\tdest.Hoge = *d\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t\tFuga: &FooPropertyInfo{\n\t\t\tfieldName: \"Fuga\",\n\t\t\tjsonName:  \"\",\n\t\t\tEncoder: func(src *Foo, dest *FooJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t} else if src.Fuga == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\td, err := NewFugaJSONBuilder().AddAll().Convert(src.Fuga)\n\t\t\t\tif err != nil {\n\t\t\t\t\treturn err\n\t\t\t\t}\n\t\t\t\tdest.FugaJSON = d\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *FooJSON, dest *Foo) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t} else if src.FugaJSON == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\td, err := src.FugaJSON.Convert()\n\t\t\t\tif err != nil {\n\t\t\t\t\treturn err\n\t\t\t\t}\n\t\t\t\tdest.Fuga = d\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t}\n\tjb._structPropertyMap[\"Tmp\"] = jb.Tmp\n\tjb._jsonPropertyMap[\"tmp\"] = jb.Tmp\n\tjb._structPropertyMap[\"Bar\"] = jb.Bar\n\tjb._jsonPropertyMap[\"\"] = jb.Bar\n\tjb._structPropertyMap[\"Buzz\"] = jb.Buzz\n\tjb._jsonPropertyMap[\"\"] = jb.Buzz\n\tjb._structPropertyMap[\"Hoge\"] = jb.Hoge\n\tjb._jsonPropertyMap[\"\"] = jb.Hoge\n\tjb._structPropertyMap[\"Fuga\"] = jb.Fuga\n\tjb._jsonPropertyMap[\"\"] = jb.Fuga\n\treturn jb\n}\n\n\nfunc (b *FooJSONBuilder) Properties() []*FooPropertyInfo {\n\treturn []*FooPropertyInfo{\n\t\tb.Tmp,\n\t\tb.Bar,\n\t\tb.Buzz,\n\t\tb.Hoge,\n\t\tb.Fuga,\n\t}\n}\n\n\nfunc (b *FooJSONBuilder) AddAll() *FooJSONBuilder {\n\tb._properties[\"Tmp\"] = b.Tmp\n\tb._properties[\"Bar\"] = b.Bar\n\tb._properties[\"Buzz\"] = b.Buzz\n\tb._properties[\"Hoge\"] = b.Hoge\n\tb._properties[\"Fuga\"] = b.Fuga\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) Add(info *FooPropertyInfo) *FooJSONBuilder {\n\tb._properties[info.fieldName] = info\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) AddByJSONNames(names ...string) *FooJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._jsonPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tb._properties[info.fieldName] = info\n\t}\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) AddByNames(names ...string) *FooJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._structPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tb._properties[info.fieldName] = info\n\t}\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) Remove(info *FooPropertyInfo) *FooJSONBuilder {\n\tdelete(b._properties, info.fieldName)\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) RemoveByJSONNames(names ...string) *FooJSONBuilder {\n\n\tfor _, name := range names {\n\t\tinfo := b._jsonPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tdelete(b._properties, info.fieldName)\n\t}\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) RemoveByNames(names ...string) *FooJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._structPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tdelete(b._properties, info.fieldName)\n\t}\n\treturn b\n}\n\n\nfunc (b *FooJSONBuilder) Convert(orig *Foo) (*FooJSON, error) {\n\tif orig == nil {\n\t\treturn nil, nil\n\t}\n\tret := &FooJSON{}\n\n\tfor _, info := range b._properties {\n\t\tif err := info.Encoder(orig, ret); err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t}\n\n\treturn ret, nil\n}\n\n\nfunc (b *FooJSONBuilder) ConvertList(orig []*Foo) (FooJSONList, error) {\n\tif orig == nil {\n\t\treturn nil, nil\n\t}\n\n\tlist := make(FooJSONList, len(orig))\n\tfor idx, or := range orig {\n\t\tjson, err := b.Convert(or)\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlist[idx] = json\n\t}\n\n\treturn list, nil\n}\n\n\nfunc (orig *FooJSON) Convert() (*Foo, error) {\n\tret := &Foo{}\n\n\tb := NewFooJSONBuilder().AddAll()\n\tfor _, info := range b._properties {\n\t\tif err := info.Decoder(orig, ret); err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t}\n\n\treturn ret, nil\n}\n\n\nfunc (jsonList FooJSONList) Convert() ([]*Foo, error) {\n\torig := ([]*FooJSON)(jsonList)\n\n\tlist := make([]*Foo, len(orig))\n\tfor idx, or := range orig {\n\t\tobj, err := or.Convert()\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlist[idx] = obj\n\t}\n\n\treturn list, nil\n}\n\n\nfunc (b *FooJSONBuilder) Marshal(orig *Foo) ([]byte, error) {\n\tret, err := b.Convert(orig)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\treturn json.Marshal(ret)\n}\n\n\ntype HogeJSON struct {\n\tHoge1 string `json:\"hoge1,omitempty\"`\n}\n\n\ntype HogeJSONList []*HogeJSON\n\n\ntype HogePropertyEncoder func(src *Hoge, dest *HogeJSON) error\n\n\ntype HogePropertyDecoder func(src *HogeJSON, dest *Hoge) error\n\n\ntype HogePropertyInfo struct {\n\tfieldName string\n\tjsonName  string\n\tEncoder   HogePropertyEncoder\n\tDecoder   HogePropertyDecoder\n}\n\n\nfunc (info *HogePropertyInfo) FieldName() string {\n\treturn info.fieldName\n}\n\n\nfunc (info *HogePropertyInfo) JSONName() string {\n\treturn info.jsonName\n}\n\n\ntype HogeJSONBuilder struct {\n\t_properties        map[string]*HogePropertyInfo\n\t_jsonPropertyMap   map[string]*HogePropertyInfo\n\t_structPropertyMap map[string]*HogePropertyInfo\n\tHoge1              *HogePropertyInfo\n}\n\n\nfunc NewHogeJSONBuilder() *HogeJSONBuilder {\n\tjb := &HogeJSONBuilder{\n\t\t_properties:        map[string]*HogePropertyInfo{},\n\t\t_jsonPropertyMap:   map[string]*HogePropertyInfo{},\n\t\t_structPropertyMap: map[string]*HogePropertyInfo{},\n\t\tHoge1: &HogePropertyInfo{\n\t\t\tfieldName: \"Hoge1\",\n\t\t\tjsonName:  \"hoge1\",\n\t\t\tEncoder: func(src *Hoge, dest *HogeJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Hoge1 = src.Hoge1\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *HogeJSON, dest *Hoge) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Hoge1 = src.Hoge1\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t}\n\tjb._structPropertyMap[\"Hoge1\"] = jb.Hoge1\n\tjb._jsonPropertyMap[\"hoge1\"] = jb.Hoge1\n\treturn jb\n}\n\n\nfunc (b *HogeJSONBuilder) Properties() []*HogePropertyInfo {\n\treturn []*HogePropertyInfo{\n\t\tb.Hoge1,\n\t}\n}\n\n\nfunc (b *HogeJSONBuilder) AddAll() *HogeJSONBuilder {\n\tb._properties[\"Hoge1\"] = b.Hoge1\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) Add(info *HogePropertyInfo) *HogeJSONBuilder {\n\tb._properties[info.fieldName] = info\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) AddByJSONNames(names ...string) *HogeJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._jsonPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tb._properties[info.fieldName] = info\n\t}\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) AddByNames(names ...string) *HogeJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._structPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tb._properties[info.fieldName] = info\n\t}\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) Remove(info *HogePropertyInfo) *HogeJSONBuilder {\n\tdelete(b._properties, info.fieldName)\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) RemoveByJSONNames(names ...string) *HogeJSONBuilder {\n\n\tfor _, name := range names {\n\t\tinfo := b._jsonPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tdelete(b._properties, info.fieldName)\n\t}\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) RemoveByNames(names ...string) *HogeJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._structPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tdelete(b._properties, info.fieldName)\n\t}\n\treturn b\n}\n\n\nfunc (b *HogeJSONBuilder) Convert(orig *Hoge) (*HogeJSON, error) {\n\tif orig == nil {\n\t\treturn nil, nil\n\t}\n\tret := &HogeJSON{}\n\n\tfor _, info := range b._properties {\n\t\tif err := info.Encoder(orig, ret); err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t}\n\n\treturn ret, nil\n}\n\n\nfunc (b *HogeJSONBuilder) ConvertList(orig []*Hoge) (HogeJSONList, error) {\n\tif orig == nil {\n\t\treturn nil, nil\n\t}\n\n\tlist := make(HogeJSONList, len(orig))\n\tfor idx, or := range orig {\n\t\tjson, err := b.Convert(or)\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlist[idx] = json\n\t}\n\n\treturn list, nil\n}\n\n\nfunc (orig *HogeJSON) Convert() (*Hoge, error) {\n\tret := &Hoge{}\n\n\tb := NewHogeJSONBuilder().AddAll()\n\tfor _, info := range b._properties {\n\t\tif err := info.Decoder(orig, ret); err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t}\n\n\treturn ret, nil\n}\n\n\nfunc (jsonList HogeJSONList) Convert() ([]*Hoge, error) {\n\torig := ([]*HogeJSON)(jsonList)\n\n\tlist := make([]*Hoge, len(orig))\n\tfor idx, or := range orig {\n\t\tobj, err := or.Convert()\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlist[idx] = obj\n\t}\n\n\treturn list, nil\n}\n\n\nfunc (b *HogeJSONBuilder) Marshal(orig *Hoge) ([]byte, error) {\n\tret, err := b.Convert(orig)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\treturn json.Marshal(ret)\n}\n\n\ntype FugaJSON struct {\n\tFuga1 string `json:\"fuga1,omitempty\"`\n}\n\n\ntype FugaJSONList []*FugaJSON\n\n\ntype FugaPropertyEncoder func(src *Fuga, dest *FugaJSON) error\n\n\ntype FugaPropertyDecoder func(src *FugaJSON, dest *Fuga) error\n\n\ntype FugaPropertyInfo struct {\n\tfieldName string\n\tjsonName  string\n\tEncoder   FugaPropertyEncoder\n\tDecoder   FugaPropertyDecoder\n}\n\n\nfunc (info *FugaPropertyInfo) FieldName() string {\n\treturn info.fieldName\n}\n\n\nfunc (info *FugaPropertyInfo) JSONName() string {\n\treturn info.jsonName\n}\n\n\ntype FugaJSONBuilder struct {\n\t_properties        map[string]*FugaPropertyInfo\n\t_jsonPropertyMap   map[string]*FugaPropertyInfo\n\t_structPropertyMap map[string]*FugaPropertyInfo\n\tFuga1              *FugaPropertyInfo\n}\n\n\nfunc NewFugaJSONBuilder() *FugaJSONBuilder {\n\tjb := &FugaJSONBuilder{\n\t\t_properties:        map[string]*FugaPropertyInfo{},\n\t\t_jsonPropertyMap:   map[string]*FugaPropertyInfo{},\n\t\t_structPropertyMap: map[string]*FugaPropertyInfo{},\n\t\tFuga1: &FugaPropertyInfo{\n\t\t\tfieldName: \"Fuga1\",\n\t\t\tjsonName:  \"fuga1\",\n\t\t\tEncoder: func(src *Fuga, dest *FugaJSON) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Fuga1 = src.Fuga1\n\t\t\t\treturn nil\n\t\t\t},\n\t\t\tDecoder: func(src *FugaJSON, dest *Fuga) error {\n\t\t\t\tif src == nil {\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t\tdest.Fuga1 = src.Fuga1\n\t\t\t\treturn nil\n\t\t\t},\n\t\t},\n\t}\n\tjb._structPropertyMap[\"Fuga1\"] = jb.Fuga1\n\tjb._jsonPropertyMap[\"fuga1\"] = jb.Fuga1\n\treturn jb\n}\n\n\nfunc (b *FugaJSONBuilder) Properties() []*FugaPropertyInfo {\n\treturn []*FugaPropertyInfo{\n\t\tb.Fuga1,\n\t}\n}\n\n\nfunc (b *FugaJSONBuilder) AddAll() *FugaJSONBuilder {\n\tb._properties[\"Fuga1\"] = b.Fuga1\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) Add(info *FugaPropertyInfo) *FugaJSONBuilder {\n\tb._properties[info.fieldName] = info\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) AddByJSONNames(names ...string) *FugaJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._jsonPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tb._properties[info.fieldName] = info\n\t}\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) AddByNames(names ...string) *FugaJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._structPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tb._properties[info.fieldName] = info\n\t}\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) Remove(info *FugaPropertyInfo) *FugaJSONBuilder {\n\tdelete(b._properties, info.fieldName)\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) RemoveByJSONNames(names ...string) *FugaJSONBuilder {\n\n\tfor _, name := range names {\n\t\tinfo := b._jsonPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tdelete(b._properties, info.fieldName)\n\t}\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) RemoveByNames(names ...string) *FugaJSONBuilder {\n\tfor _, name := range names {\n\t\tinfo := b._structPropertyMap[name]\n\t\tif info == nil {\n\t\t\tcontinue\n\t\t}\n\t\tdelete(b._properties, info.fieldName)\n\t}\n\treturn b\n}\n\n\nfunc (b *FugaJSONBuilder) Convert(orig *Fuga) (*FugaJSON, error) {\n\tif orig == nil {\n\t\treturn nil, nil\n\t}\n\tret := &FugaJSON{}\n\n\tfor _, info := range b._properties {\n\t\tif err := info.Encoder(orig, ret); err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t}\n\n\treturn ret, nil\n}\n\n\nfunc (b *FugaJSONBuilder) ConvertList(orig []*Fuga) (FugaJSONList, error) {\n\tif orig == nil {\n\t\treturn nil, nil\n\t}\n\n\tlist := make(FugaJSONList, len(orig))\n\tfor idx, or := range orig {\n\t\tjson, err := b.Convert(or)\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlist[idx] = json\n\t}\n\n\treturn list, nil\n}\n\n\nfunc (orig *FugaJSON) Convert() (*Fuga, error) {\n\tret := &Fuga{}\n\n\tb := NewFugaJSONBuilder().AddAll()\n\tfor _, info := range b._properties {\n\t\tif err := info.Decoder(orig, ret); err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t}\n\n\treturn ret, nil\n}\n\n\nfunc (jsonList FugaJSONList) Convert() ([]*Fuga, error) {\n\torig := ([]*FugaJSON)(jsonList)\n\n\tlist := make([]*Fuga, len(orig))\n\tfor idx, or := range orig {\n\t\tobj, err := or.Convert()\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlist[idx] = obj\n\t}\n\n\treturn list, nil\n}\n\n\nfunc (b *FugaJSONBuilder) Marshal(orig *Fuga) ([]byte, error) {\n\tret, err := b.Convert(orig)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\treturn json.Marshal(ret)\n}
"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

print(len(input_ids[0]))

# simply generate a single sequence
# generated_ids = model.generate(input_ids, max_length=8)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
