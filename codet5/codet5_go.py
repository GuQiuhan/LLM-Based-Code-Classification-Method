from transformers import T5ForConditionalGeneration, RobertaTokenizer

# load model and tokenizer
model_path = "PPY039/codet5-small-go_generation_v2"

model = T5ForConditionalGeneration.from_pretrained(model_path, cache_dir="D:\huggingface_cache")
tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir="D:\huggingface_cache")

# use model
input_text = "package names\n\nimport \"knative.dev/pkg/kmeta\"\n\n\nfunc Deployment(rev kmeta.Accessor) string {\n\treturn kmeta.ChildName(rev.GetName(), \"-deployment\")\n}\n\n\nfunc ImageCache(rev kmeta.Accessor) string {\n\treturn kmeta.ChildName(rev.GetName(), \"-cache\")\n}\n\n\n\n\nfunc PA(rev kmeta.Accessor) string"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids=input_ids, max_new_tokens=256)  # max_trg_len = 256

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# this prints "{\n\treturn kmeta.ChildName(rev.GetName(), "-pa")\n}"
print(output_text)



