import json
import torch


def get_cat_text_label(processor, follow_up=""):
    template = "answer: <cat>" + follow_up
    cat1_text = template.replace("<cat>", "cat_1")
    cat2_text = template.replace("<cat>", "cat_2")
    cat1_tokens = processor(text=cat1_text)
    cat2_tokens = processor(text=cat2_text)

    longest_label = 0
    for cat in [cat1_tokens, cat2_tokens]:
        if isinstance(cat["input_ids"], list):
            cat["input_ids"] = torch.tensor(cat["input_ids"])
        if len(cat["input_ids"].shape) == 1:
            cat["input_ids"] = cat["input_ids"].unsqueeze(0)
        cat["input_ids"] = cat["input_ids"][:, 1:] # idk how to skip adding start tokens, so filter em out
        longest_label = max(len(cat["input_ids"][0]), longest_label)

    cat_text = {"cat_1": cat1_text, "cat_2": cat2_text}
    cat_labels = {"cat_1": cat1_tokens["input_ids"].cuda(), "cat_2": cat2_tokens["input_ids"].cuda()}
    return cat_text, cat_labels, longest_label


def get_urls_and_cat_from_item(item, dataset_path="../"):
    cat2_imgs = item["imagefiles"]["cat_2"]
    cat1_imgs = item["imagefiles"]["cat_1"]
    test_img = item["testfiles"]["testimage"]
    test_cat = item["testfiles"]["category"]
    urls = cat2_imgs + cat1_imgs + [test_img]
    urls = [dataset_path + x for x in urls]
    return urls, test_cat


def get_ds(dataset, train_limit=100000, val_limit=10000):
    if dataset == "openworld":
        with open("../bongard_openworld.json", 'r') as f:
            data = json.load(f)
        train, validate, test = data[501:1001], [("val", data[1001:1101])], [("test", data[0:500])]

        overlap = lambda a, b: set([x["uid"] for x in a]) & set([x["uid"] for x in b])
        print(f"overlap of train and validate: {overlap(train, validate[0][1])}")
        print(f"overlap of train and test: {overlap(train, test[0][1])}")
        print(f"overlap of validate and test: {overlap(test[0][1], validate[0][1])}")
    elif dataset == "hoi":
        train, validate, test = [], [], []
        with open("../bongard_hoi.json", 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if key == "train":
                items = min(train_limit, 4000)
                train.extend(value[:items])
            if key.startswith("val"):
                items = min(val_limit, 100)
                validate.append((key, value[:items]))
            if key.startswith("test"):
                test.append((key, value[:200]))
                distribution = {}
                for item in value[:200]:
                    concept = item["concept"]
                    if concept not in distribution:
                        distribution[concept] = 0
                    distribution[concept] += 1
                print("test distribution: ", distribution)
    else:
        raise Exception("Unknown dataset")

    print(f"len train={len(train)}, len validate={[len(x[1]) for x in validate]} len test={[len(x[1]) for x in test]}")
    return train, validate, test


direct_prompt = (f"Your task is to:\n"
                 f"1. Provide your conclusion for the `test image` if it can be categorized as either `cat_1` or `cat_2` based on the analysis and the rule.\n"
                 f"\n"
                 f"The format of your output should be as follows:\n"
                 f"Conclusion: cat_1 or cat_2\n\nConclusion should be 1 category only!")

cot_prompt = (f"Your task is to:\n"
              f"1. Determine the rule or criterion that distinguishes the `cat_2` samples from the `cat_1` ones.\n"
              f"2. Analyse the `test image`.\n"
              f"3. Provide your conclusion for the `test image` if it can be categorized as either `cat_1` or `cat_2` based on the analysis and the rule.\n"
              f"\n"
              f"Ensure that the output is clear, well-formatted, and free of unnecessary explanations, and no newline after : \n"
              f"The format of your output should be as follows:\n"
              f"Analysis: (Your analysis here)\n"
              f"Rule: (The distinguishing rule here)\n"
              f"Test Image: (Test image details)\n"
              f"Conclusion: cat_1 or cat_2\n\nConclusion should be 1 category only!")


def interleaved_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img):
    return (f"You are presented a bongard task. There are {image_count} pictures total. \n"
            f"First {cat_imgs} samples belong to cat_2, which follow 1 common rule. Here they are: {cat2_imgs} . \n"
            f"Following {cat_imgs} distinctly do not follow that rule and are cat_1. Here they are: {cat1_imgs} . \n"
            f"Last image is a test image you need to categorize either as cat_2 or cat_1 based on the rule, which is here: {test_img} . \n"
            f"If test image follows the rule, it's cat_2. If it doesn't follow the rule, it's cat_1. \n")


def interleaved_test_first_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img):
    return (f"You are presented a bongard task. There are {image_count} pictures total. \n"
            f"First image is a test image you need to categorize either as cat_2 or cat_1 based on the rule. \n"
            f"Here is a test image: {test_img}. \n"
            f"{cat_imgs} samples belong to cat_2, which follow 1 common rule. Here they are: {cat2_imgs} . \n"
            f"Following {cat_imgs} distinctly do not follow that rule and are cat_1. Here they are: {cat1_imgs} .\n"
            f"If test image follows the rule, it's cat_2. If it doesn't follow the rule, it's cat_1. \n")

def labeled_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img):
    return (f"You are presented a bongard task. There are {image_count} pictures total. First {cat_imgs} samples belong to cat_2, which follow 1 common rule.\n"
            f"Following {cat_imgs} distinctly do not follow that rule and are cat_1.\n"
            f"Last image is a test image you need to categorize either as cat_2 or cat_1 based on the rule. \n"
            f"If test image follows the rule, it's cat_2. If it doesn't follow the rule, it's cat_1. \n"
            f"Here are the cat2 images: {cat2_imgs} , cat1 images: {cat1_imgs} , test image: {test_img} .\n")

def labeled_test_first_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img):
    return (f"You are presented a bongard task. There are {image_count} pictures total. \n"
            f"First image is a test image you need to categorize either as cat_2 or cat_1 based on the rule. \n"
            f"Following {cat_imgs} samples belong to cat_2, which follow 1 common rule.\n"
            f"Last {cat_imgs} distinctly do not follow that rule and are cat_1.\n"
            f"If test image follows the rule, it's cat_2. If it doesn't follow the rule, it's cat_1. \n"
            f"Here is the test image: {test_img} , cat2 images: {cat2_imgs} , cat1 images: {cat1_imgs} .\n")


### PEFT ###

def get_phi_simple_prompt(image_count, base_prompt=interleaved_prompt):
    cat_imgs = (image_count - 1) // 2
    cat2_imgs = "".join([f"\n<|image_{i}|>" for i in range(1, cat_imgs + 1)])
    cat1_imgs = "".join([f"\n<|image_{i}|>" for i in range(cat_imgs + 1, cat_imgs * 2 + 1)])
    test_img = f"\n<|image_{image_count}|>"
    prompt = (f"{base_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}"
              "Return the classification. format='answer: <cat>'.\n")
    return f"<|user|> {prompt} <|end|>\n<|assistant|>\n"

def get_phi_trainable_prompt(image_count, trainable_tokens, base_prompt=interleaved_prompt):
    cat_imgs = (image_count - 1) // 2
    cat2_imgs = "".join([f"\n<|image_{i}|>" for i in range(1, cat_imgs + 1)])
    cat1_imgs = "".join([f"\n<|image_{i}|>" for i in range(cat_imgs + 1, cat_imgs * 2 + 1)])
    test_img = f"\n<|image_{image_count}|>"
    preamble = "Hello. I have a task for you. A very important task. "
    general_instruction = (f"General instruction: \n{base_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}"
                           "Also construct a specific instruction which you think will help the most for this kind of task. "
                           "Think of an algorithm to follow, what kind of components to look for on the images, how they could be combined to follow some rule. Be creative! :D "
                           "Return the classification before anything else and then the the specific instruction word-by-word and nothing else!\n\n")
    simplified_general_instruction = f"cat_2 images: {cat2_imgs}. cat_1 images: {cat1_imgs}. test image: {test_img}. return 'answer: <cat_1 or cat_2>'!"
    specific_instruction = "Here is the specific instruction which helps you a lot: \n"
    specific_instruction_preamble = "With all things considered. "
    specific_naturalized_instruction = ("Here is the specific instruction which helps you a lot: \n"
                                        "1. Identify the category of each image by examining the content and context. "
                                        "2. For images in cat_2, determine if they share a common rule or theme. "
                                        "3. For images in cat_1, identify what distinguishes them from the cat_2 images. "
                                        "4. For the test image, apply the same rules to decide its category. "
                                        "5. Provide the category and a brief explanation for each image based on the identified rules. \n"
                                        "Make sure to return the answer in 'answer: <cat>, explanation:' format!")
    prompt = f"<|user|> {general_instruction} {specific_instruction} {'<unk>' * trainable_tokens} <|end|>\n<|assistant|>\n"
    return prompt


def get_phi_trainable_prompt_full(image_count, trainable_tokens, base_prompt=interleaved_prompt):
    return f"{'<unk>' * trainable_tokens}{get_phi_trainable_prompt(image_count, trainable_tokens, base_prompt=base_prompt)}"


def get_pixtral_simple_prompt(image_count, base_prompt=interleaved_prompt):
    cat_imgs = (image_count - 1) // 2
    cat2_imgs = "".join(["[IMG]"] * cat_imgs)
    cat1_imgs = "".join(["[IMG]"] * cat_imgs)
    test_img = f"[IMG]"
    prompt = (f"{base_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}"
              "Return the classification. format='answer: <cat>'.\n")
    return f"<s>[INST] {prompt} [/INST]"



def get_pixtral_trainable_prompt(image_count, trainable_tokens, base_prompt=interleaved_prompt):
    cat_imgs = (image_count - 1) // 2
    cat2_imgs = "".join(["[IMG]"] * cat_imgs)
    cat1_imgs = "".join(["[IMG]"] * cat_imgs)
    test_img = f"[IMG]"
    general_instruction = (f"General instruction: \n{base_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}"
                           "Also construct a specific instruction which you think will help the most for this kind of task. "
                           "Think of an algorithm to follow, what kind of components to look for on the images, how they could be combined to follow some rule. Be creative! :D "
                           "Return the classification before anything else and then the the specific instruction word-by-word and nothing else!\n\n")
    specific_instruction = "Here is the specific instruction which helps you a lot: \n"
    prompt = f"<s>[INST] {general_instruction} {specific_instruction} {'<unk>' * trainable_tokens} [/INST]"
    return prompt


def get_pixtral_trainable_prompt_full(image_count, trainable_tokens, base_prompt=interleaved_prompt):
    return f"{'<unk>' * trainable_tokens}{get_pixtral_trainable_prompt(image_count, trainable_tokens, base_prompt=base_prompt)}"


def get_gemma_simple_prompt(image_count, base_prompt=interleaved_prompt):
    cat_imgs = (image_count - 1) // 2
    cat2_imgs = "".join(["<start_of_image>"] * cat_imgs)
    cat1_imgs = "".join(["<start_of_image>"] * cat_imgs)
    test_img = "<start_of_image>"
    prompt = (f"{base_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}"
              "Return the classification. format='answer: <cat>'.\n")
    return f"<bos><start_of_turn>user\n{prompt} <end_of_turn>\n<start_of_turn>model\n"



def get_gemma_trainable_prompt(image_count, trainable_tokens, base_prompt=interleaved_prompt):
    cat_imgs = (image_count - 1) // 2
    cat2_imgs = "".join(["<start_of_image>"] * cat_imgs)
    cat1_imgs = "".join(["<start_of_image>"] * cat_imgs)
    test_img = "<start_of_image>"
    general_instruction = (f"General instruction: \n{base_prompt(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}"
                           "Also construct a specific instruction which you think will help the most for this kind of task. "
                           "Think of an algorithm to follow, what kind of components to look for on the images, how they could be combined to follow some rule. Be creative! :D "
                           "Return the classification before anything else and then the the specific instruction word-by-word and nothing else!\n\n")
    specific_instruction = "Here is the specific instruction which helps you a lot: \n"
    prompt = f"<bos><start_of_turn>user\n{general_instruction} {specific_instruction} {'<unk>' * trainable_tokens} <end_of_turn>\n<start_of_turn>model\n"
    return prompt


def get_gemma_trainable_prompt_full(image_count, trainable_tokens, base_prompt=interleaved_prompt):
    return f"{'<unk>' * trainable_tokens}{get_gemma_trainable_prompt(image_count, trainable_tokens, base_prompt=base_prompt)}"



### Baselines ###

def construct_prompt_phi(cat_imgs, image_count, prompt_method, prompt_structure, test_first, is_direct):
    if test_first:
        test_img = f"\n<|image_{1}|>"
        cat2_imgs = "".join([f"\n<|image_{i}|>" for i in range(2, cat_imgs + 2)])
        cat1_imgs = "".join([f"\n<|image_{i}|>" for i in range(cat_imgs + 2, cat_imgs * 2 + 2)])
    else:
        cat2_imgs = "".join([f"\n<|image_{i}|>" for i in range(1, cat_imgs + 1)])
        cat1_imgs = "".join([f"\n<|image_{i}|>" for i in range(cat_imgs + 1, cat_imgs * 2 + 1)])
        test_img = f"\n<|image_{image_count}|>"
    appendix = "Conclusion:" if is_direct else ""
    full_prompt = f"<|user|> {prompt_structure(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}{prompt_method} <|end|>\n<|assistant|>\n{appendix}"
    return full_prompt

def construct_prompt_pixtral(cat_imgs, image_count, prompt_method, prompt_structure, test_first, is_direct):
    test_img = "[IMG]"
    cat2_imgs = "".join(["[IMG]"] * cat_imgs)
    cat1_imgs = "".join(["[IMG]"] * cat_imgs)
    appendix = "Conclusion: " if is_direct else ""
    full_prompt = f"<s>[INST] {prompt_structure(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}{prompt_method} [/INST]\n{appendix}"
    return full_prompt

def construct_prompt_gemma(cat_imgs, image_count, prompt_method, prompt_structure, test_first, is_direct):
    test_img = "<start_of_image>"
    cat2_imgs = "".join(["<start_of_image>"] * cat_imgs)
    cat1_imgs = "".join(["<start_of_image>"] * cat_imgs)
    appendix = "Conclusion:" if is_direct else ""
    full_prompt = f"<bos><start_of_turn>user\n{prompt_structure(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}{prompt_method} <end_of_turn>\n<start_of_turn>model\n{appendix}"
    return full_prompt

def construct_prompt_intern(cat_imgs, image_count, prompt_method, prompt_structure, test_first, is_direct):
    test_img = "<image>"
    cat2_imgs = "".join(["<image>"] * cat_imgs)
    cat1_imgs = "".join(["<image>"] * cat_imgs)
    appendix = "Conclusion" if is_direct else ""
    full_prompt = f"User: {prompt_structure(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}{prompt_method}\nAssistant: {appendix}"
    return full_prompt

def construct_prompt_qwen(cat_imgs, image_count, prompt_method, prompt_structure, test_first, is_direct):
    test_img = "<|vision_start|><|image_pad|><|vision_end|>"
    cat2_imgs = "".join(["<|vision_start|><|image_pad|><|vision_end|>"] * cat_imgs)
    cat1_imgs = "".join(["<|vision_start|><|image_pad|><|vision_end|>"] * cat_imgs)
    appendix = "Conclusion: " if is_direct else ""
    full_prompt = f"<|im_start|>user\n{prompt_structure(cat1_imgs, cat2_imgs, cat_imgs, image_count, test_img)}{prompt_method}<|im_end|>\n<|im_start|>assistant {appendix}"
    return full_prompt
