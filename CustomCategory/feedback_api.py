import json
import openai
import random 


class AutoMitigationClient():
    def __init__(self, **args) -> None:
        self.endpoint = args.get("endpoint")
        self.headers = args.get("headers")
        self.api_key = args.get("openai_api_key")
        self.api_base = args.get("openai_api_base")
        self.api_type = args.get("openai_api_type")
        self.api_version = args.get("openai_api_version")
        self.deploy = args.get("openai_api_deploy")

        self.dsat_false_negative = None
        self.dsat_false_positive = None

    def feedback_reasoning(self, guideline, false_positives=None, false_negatives=None):
        result = {"FP": None, "FN": None}
        if false_negatives != None:
            guideline["misclassified_texts"] = false_negatives
            self.dsat_false_negative = false_negatives
            messages = [
                {
                    "role": "system", "content": 
                    """
                    You are an online content moderator who has developed an online classifier to detect harmful content within a specific category. You have been informed about the category of harmful content and given a detailed guide outlining what type of text falls within this category and what does not. A concise category description and relevant sample data are used for data augmentation, and with this augmented data, you have trained the classifier. This model is able to determine whether a given text belongs to the harmful category or not. However, the model is not performing well on the evaluation dataset, incorrectly labeling some harmful texts as safe. We require your assistance in improving the model's performance. You need to examine why the model misclassified the text and investigate possible solutions to this problem, potentially from the following perspectives:

                    1. If the sample matches the description of the safe category, it suggests an error in the description that led the model to mistakenly classify the sample as safe. In this case, you will need to modify the safe description.
                    2. If the sample does not match either the harmful or safe description, you need to add a specific description of this sample to the harmful description, allowing the model to encounter more of this type of data during training and thus avoid similar issues.
                    3. If the sample matches the harmful category description, but the description is too broad and doesn't specifically address this type of sample, it fails to generate the corresponding training set, leaving the model unable to determine if the sample is safe.
                    4. If the sample matches the harmful category description and the description also provides a specific illustration, it is mistakenly classified as safe because some keywords that frequently appear in the safe training data lead the model to mistakenly deem the keyword as safe. You need to identify which keyword leads to the incorrect detection.

                    You will be provided with the description, samples, and misclassified text in the form below:
                    {"category": "The harmful content's category name", "harmful_description": "A clear guideline about what kind of text is considered harmful", "safe_description":"Description of safe data", "misclassified_texts": ["Cases that are incorrectly classified as safe but should be considered harmful data", ... ]}

                    After identifying the cause, you must explain it and generate new sample data based on your reasoning for us to test the accuracy of your diagnosis.

                    Your response should be in JSON format as shown below: \n
                    {"reason": "", "generated_data": ["related harmful samples", "", ...]}
                    """
                },
                {
                    "role": "user", "content": json.dumps(guideline)
                },
            ]
            while True:
                try:
                    print("FN feedback reasoning start")
                    response = openai.ChatCompletion.create(
                        engine=self.deploy,
                        messages=messages,
                        temperature=0.7,
                        timeout=450,
                    )
                    print(json.loads(response["choices"][0]["message"]["content"]))
                    result["FN"] = json.loads(response["choices"][0]["message"]["content"])
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"FN feedback reasoning error: {e}")
                    raise
                    # continue
        if false_positives != None:
            guideline["misclassified_texts"] = false_positives
            self.dsat_false_positive = false_positives
            messages = [
                {
                    "role": "system", "content": 
                    """
                    You are an online content moderator who has developed an online classifier to detect harmful content within a specific category. You have been informed about the category of harmful content and given a detailed guide outlining what type of text falls within this category and what does not. A concise category description and relevant sample data are used for data augmentation, and with this augmented data, you have trained the classifier. This model is able to determine whether a given text belongs to the harmful category or not. However, the model is not performing well on the evaluation dataset, incorrectly labeling some safe texts as harmful. We require your assistance in improving the model's performance. You need to examine why the model misclassified the text and investigate possible solutions to this problem, potentially from the following perspectives:

                    1. If the sample matches the description of the harmful category, it suggests an error in the description that led the model to mistakenly classify the sample as harmful. In this case, you will need to modify the harmful description.
                    2. If the sample does not match either the harmful or safe description, you need to add a specific description of this sample to the safe description, allowing the model to encounter more of this type of data during training and thus avoid similar issues.
                    3. If the sample matches the safe category description, but the description is too broad and doesn't specifically address this type of sample, it fails to generate the corresponding training set, leaving the model unable to determine if the sample is safe.
                    4. If the sample matches the safe category description and the description also provides a specific description, it is mistakenly classified as harmful because some keywords that frequently appear in the harmful training data lead the model to mistakenly deem the keyword as harmful. You need to identify which keyword leads to the incorrect detection.

                    You will be provided with the description, samples, and misclassified text in the form below:
                    {"category": "The harmful content's category name", "harmful_description": "A clear guideline about what kind of text is considered harmful", "safe_description":"Description of safe data", "harmful_sample_data": ["Sample data of the harmful category used as seeds for data augmentation", ...], "safe_sample_data": ["Sample data of the safe category used as seeds for data augmentation", ...]", "misclassified_texts": ["Cases that are incorrectly classified as harmful but should be considered safe data", ... ]}

                    After identifying the cause, you must explain it and generate new sample data based on your reasoning for us to test the accuracy of your diagnosis.

                    Your response should be in JSON format as shown below: \n
                    {"reason": "", "generated_data": ["related safe samples", "", ...]}
                    """
                },
                {
                    "role": "user", "content": json.dumps(guideline)
                },
            ]
            while True:
                try:
                    print("FP feedback reasoning start")
                    response = openai.ChatCompletion.create(
                        engine=self.deploy,
                        messages=messages,
                        temperature=0.7,
                        timeout=450,
                    )
                    print(response["choices"][0]["message"]["content"])
                    result["FP"] = json.loads(response["choices"][0]["message"]["content"])
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"FP feedback reasoning error: {e}")
                    raise
        return result

    def edit_description(self, guideline, assist_last):
        result = {"edited": False, "harmful_description": None, "safe_description": None}
        if assist_last["FN"] != None:
            guideline["misclassified_texts"] = self.dsat_false_negative
            messages = [
                {
                    "role": "system", "content": 
                    """
                    You are an online content moderator who has developed an online classifier to detect harmful content within a specific category. You have been informed about the category of harmful content and given a detailed guide outlining what type of text falls within this category and what does not. A concise category description and relevant sample data are used for data augmentation, and with this augmented data, you have trained the classifier. This model is able to determine whether a given text belongs to the harmful category or not. However, the model is not performing well on the evaluation dataset, incorrectly labeling some harmful texts as safe. We require your assistance in improving the model's performance. You need to examine why the model misclassified the text and investigate possible solutions to this problem, potentially from the following perspectives:

                    1. If the sample matches the description of the safe category, it suggests an error in the description that led the model to mistakenly classify the sample as safe. In this case, you will need to modify the safe description.
                    2. If the sample does not match either the harmful or safe description, you need to add a specific description of this sample to the harmful description, allowing the model to encounter more of this type of data during training and thus avoid similar issues.
                    3. If the sample matches the harmful category description, but the description is too broad and doesn't specifically address this type of sample, it fails to generate the corresponding training set, leaving the model unable to determine if the sample is safe.
                    4. If the sample matches the harmful category description and the description also provides a specific description, it is mistakenly classified as safe because some keywords that frequently appear in the harmful training data lead the model to mistakenly deem the keyword as harmful. You need to identify which keyword leads to the incorrect detection.

                    You will be provided with the description, samples, and misclassified text in the form below:
                    {"category": "The harmful content's category name", "harmful_description": "A clear guideline about what kind of text is considered harmful", "safe_description":"Description of safe data",  "misclassified_texts": ["Cases that are incorrectly classified as safe but should be considered harmful data", ... ]}

                    After identifying the cause, you must explain it and generate new sample data based on your reasoning for us to test the accuracy of your diagnosis.

                    Your response should be in JSON format as shown below: \n
                    {"reason": "", "generated_data": ["related harmful samples", "", ...]}
                    """
                },
                {
                    "role": "user", "content": json.dumps(guideline)
                },
                {
                    "role": "assistant", "content": json.dumps(assist_last)
                },
                {
                    "role": "system", "content": "Your reasoning is correct, to train a better model, you need to decide if the false negative is caused by the existing descripiton is not good enough. If you think modify the description can resolve the problem, you need to ensure your modifications do not negatively impact the model's existing performance. Succinctly add the identified issues within the description, only deleting previous content if you are certain it is incorrect. And **remember that your modification is to mitigate all related issues, do not only focus on words in the given samples** \n .Only response should be in JSON format as shown below: \n {\"edited\": true/false(whether the description is modified), \"harmful_description\": \"The modified harmful description\", \"safe_description\": \"The modified safe description\", \"reasoning\": \"your reason for the modification\"}"
                }
            ]
            while True:
                try:
                    response_fn = openai.ChatCompletion.create(
                        engine=self.deploy,
                        messages=messages,
                        temperature=0.7,
                        timeout=450,
                    )
                    response_fn = json.loads(response_fn["choices"][0]["message"]["content"])
                    if response_fn["edited"]:
                        guideline["harmful_description"] = response_fn["harmful_description"]
                        guideline["safe_description"] = response_fn["safe_description"]
                        print(f"FN description edited: {response_fn["reasoning"]}")
                        result["edited"] = True
                        result["harmful_description"] = response_fn["harmful_description"]
                        result["safe_description"] = response_fn["safe_description"]
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"FN description edit error: {e}")
                    continue
        if assist_last["FP"] != None:
            guideline["misclassified_texts"] = self.dsat_false_positive
            messages = [
                {
                    "role": "system", "content": 
                    """
                    You are an online content moderator who has developed an online classifier to detect harmful content within a specific category. You have been informed about the category of harmful content and given a detailed guide outlining what type of text falls within this category and what does not. A concise category description and relevant sample data are used for data augmentation, and with this augmented data, you have trained the classifier. This model is able to determine whether a given text belongs to the harmful category or not. However, the model is not performing well on the evaluation dataset, incorrectly labeling some safe texts as harmful. We require your assistance in improving the model's performance. You need to examine why the model misclassified the text and investigate possible solutions to this problem, potentially from the following perspectives:

                    1. If the sample matches the description of the harmful category, it suggests an error in the description that led the model to mistakenly classify the sample as harmful. In this case, you will need to modify the harmful description.
                    2. If the sample does not match either the harmful or safe description, you need to add a specific description of this sample to the safe description, allowing the model to encounter more of this type of data during training and thus avoid similar issues.
                    3. If the sample matches the safe category description, but the description is too broad and doesn't specifically address this type of sample, it fails to generate the corresponding training set, leaving the model unable to determine if the sample is safe.
                    4. If the sample matches the safe category description and the description also provides a specific description, it is mistakenly classified as harmful because some keywords that frequently appear in the harmful training data lead the model to mistakenly deem the keyword as harmful. You need to identify which keyword leads to the incorrect detection.

                    You will be provided with the description, samples, and misclassified text in the form below:
                    {"category": "The harmful content's category name", "harmful_description": "A clear guideline about what kind of text is considered harmful", "safe_description":"Description of safe data", "misclassified_texts": ["Cases that are incorrectly classified as harmful but should be considered safe data", ... ]}

                    After identifying the cause, you must explain it and generate new sample data based on your reasoning for us to test the accuracy of your diagnosis.

                    Your response should be in JSON format as shown below: \n
                    {"reason": "", "generated_data": ["related safe samples", "", ...]}
                    """
                },
                {
                    "role": "user", "content": json.dumps(guideline)
                },
                {
                    "role": "assistant", "content": json.dumps(assist_last)
                },
                {
                    "role": "system", "content": "Your reasoning is correct, to train a better model, you need to decide if the false positive is caused by the existing descripiton is not good enough. If you think modify the description can resolve the problem, you need to ensure your modifications do not negatively impact the model's existing performance. Succinctly add the identified issues within the description, only deleting previous content if you are certain it is incorrect. And **remember that your modification is to mitigate all related issues, do not only focus on words in the given samples** \n\n .Only response should be in JSON format as shown below: \n {\"edited\": true/false(whether the description is modified), \"harmful_description\": \"The modified harmful description\", \"safe_description\": \"The modified safe description\", \"reasoning\": \"your reason for the modification\"}"
                }
            ]
            while True:
                try:
                    response_fp = openai.ChatCompletion.create(
                        engine=self.deploy,
                        messages=messages,
                        temperature=0.7,
                        timeout=450,
                    )
                    response_fp = json.loads(response_fp["choices"][0]["message"]["content"])
                    if response_fp["edited"]:
                        print(f"FP description edited: {response_fp["reasoning"]}")
                        result["edited"] = True
                        result["harmful_description"] = response_fp["harmful_description"]
                        result["safe_description"] = response_fp["safe_description"]
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"FP description edit error: {e}")
                    continue
        return result

    def augmentation_fn(self, guideline, samples):
        messages = [
            {
                "role": "system", "content": "You are an imaginative WriterGPT, we are training a model to identify harmful content for a specific category, however we don't have enough data to train the model. We need your ability to generate diverse"
                "training data. You need to generate 5 different harmful texts and 5 normal texts. \n "
                "Having a lot of similar data in the training set can lead to overfitting, which results in poor performance, we need your talent to generate diverse "
                "data. Below are advices for you to generate diverse data: \n "
                "  1. Writing Style: Vary the writing style, including narrative, descriptive, expository, argumentative, and poetic styles.\n"
                "  2. Text Type: Explore different text types such as prose, essays, reports, dialogues, and poetry.\n"
                "  3. Tone and Emotion: Modify the tone and emotional context of the text, ranging from formal to informal, and express different emotions like joy, anger, sadness, etc.\n"
                "  4. Theme and Domain: Augment data for different themes and domains, covering various subjects like news, technology, healthcare, sports, etc.\n"
                "  5. Cultural and Regional Variations: Incorporate diverse cultural and regional variations by using specific terminologies, expressions, and idioms relevant to different cultures and regions.\n"
                "  6. Length and Complexity: Generate texts with varying lengths and complexity, including both short and concise pieces as well as longer and more intricate texts.\n"
                "  7. Language and Style: Include different languages and writing styles by translating texts, utilizing specific languages, or adopting different styles such as formal, humorous, academic, colloquial, etc. \n \n  "
                "During the generation process, because you will generate 10 different texts (5 harmful texts and 5 not harmful texts), for each text, you should select different strategies above to get generated data in different styles! "
                "In addition to that, I will also provide you with some feedbacks, you can refer to them to generate your own data. But remember, ** do not ** generate texts that the same to the given examples. And beside generating the harmful content, I need you to generate negative data for training. The pattern of the negative data should be the same to the harmful content with the same pattern and wording. The reason for you to do that is to avoid model overfitting, because the positive samples that you generate would share the same pattern. So I need you to write negative samples with the same pattern but the content is not harmful \n"
                f"Your result should be based on a random seed, different random seeds will generate different results. Now your random seed is {random.randint(0, 1000000)} \n"
            },
            {
                "role": "system", "content": "You should generate diverse training data. Same content in the result is not allowed. Generate Only respond in JSON format as described below, do not use words like \"and so on\" or \"...\" for omission, "\
                "and **Do not number the generated data like: \n [\"1. content\", \"2. ...]** \n \n Response Format should be in the format below:"
                " \n {\"harmful\": [\"content1\", \"content2\", ...], \"normal\": [\"content1\", \"content2\", ...]}"
            },
            {
                "role": "system", "content": "Category information in format as described below, \Input Format:"
                " \n {\"category\": \"name\", \"description\": \"description\", \"feedbacks\": [\"feedback1\", \"feedback2\", ...]}"
            },
            {
                "role": "user", "content":  json.dumps({"category": guideline["category"], "description": guideline["harmful_description"], "feedbacks": samples})
            }
        ]
        while True:
            try:
                response = openai.ChatCompletion.create(
                    engine=self.deploy,
                    messages=messages,
                    temperature=0.7,
                    timeout=450,
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"FN augmentation error: {e}")
                continue
        return json.loads(response["choices"][0]["message"]["content"])["harmful"]
    
    def augmentation_fp(self, guideline, samples):
        messages = [
            {
                "role": "system", "content": "You are an imaginative WriterGPT, we are training a model to identify harmful content for a specific category, however we don't have enough data to train the model. We need your ability to generate diverse"
                "training data. You need to generate 5 different normal texts to mitigate the false positive cases that we meet. \n "
                "Having a lot of similar data in the training set can lead to overfitting, which results in poor performance, we need your talent to generate diverse "
                "data. Below are advices for you to generate diverse data: \n "
                "  1. Writing Style: Vary the writing style, including narrative, descriptive, expository, argumentative, and poetic styles.\n"
                "  2. Text Type: Explore different text types such as prose, essays, reports, dialogues, and poetry.\n"
                "  3. Tone and Emotion: Modify the tone and emotional context of the text, ranging from formal to informal, and express different emotions like joy, anger, sadness, etc.\n"
                "  4. Theme and Domain: Augment data for different themes and domains, covering various subjects like news, technology, healthcare, sports, etc.\n"
                "  5. Cultural and Regional Variations: Incorporate diverse cultural and regional variations by using specific terminologies, expressions, and idioms relevant to different cultures and regions.\n"
                "  6. Length and Complexity: Generate texts with varying lengths and complexity, including both short and concise pieces as well as longer and more intricate texts.\n"
                "  7. Language and Style: Include different languages and writing styles by translating texts, utilizing specific languages, or adopting different styles such as formal, humorous, academic, colloquial, etc. \n \n  "
                "During the generation process, because you will generate 5 different normal texts, for each text, you should select different strategies above to get generated data in different styles! "
                "In addition to that, you will be provided with some examples, you can refer to them to generate your own data. But remember, ** do not ** generate texts that the same to the given examples. And beside generating the normal content. "
                f"Your result should be based on a random seed, different random seeds will generate different results. Now your random seed is {random.randint(0, 1000000)} \n"
            },
            {
                "role": "system", "content": "You should generate diverse training data. Same content in the result is not allowed. Generate Only respond in JSON format as described below, do not use words like \"and so on\" or \"...\" for omission, "\
                "and **Do not number the generated data like: \n [\"1. content\", \"2. ...]** \n \n Response Format should be in the format below:"
                " \n [\"normal content1\", \"normal content2\", ...]"
            },
            {
                "role": "system", "content": "Category information is as described below, \n Input Format:"
                " \n {\"category\": \"harmful category name\", \"harmful_description\": \"harmful description\", \"safe_description\": \"safe description\", \"safe_samples\": [\"sample1\", \"sample2\", ...]}"
            },
            {
                "role": "user", "content":  json.dumps({"category": guideline["category"], "harmful_description": guideline["harmful_description"], "safe_description": guideline["safe_description"], "safe_samples": samples})
            }
        ]
        while True:
            try:
                response = openai.ChatCompletion.create(
                    engine="gpt-4-32k",
                    messages=messages,
                    temperature=0.7,
                    timeout=450,
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"FP augmentation error: {e}")
                continue
        return json.loads(response["choices"][0]["message"]["content"])