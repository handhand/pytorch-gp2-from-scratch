#### pytorch实现GPT2

参考这本书：[Build a Large Language Model (From Scratch)](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167/ref=sr_1_1?crid=3B4BFSK81NSPY&dib=eyJ2IjoiMSJ9.-QLY_LozQcjAJ1ZcSFAlzQfH33M2v9H4_9H60MePz1ihr3RiFBqraejs4A590XmeSOb3iKq5xR9QD8gd5rbGnRYWapP--iEy1CfGW8NLUvBi5UaR6Nqcbmm-Vy6hvhnUEhAnGzv10VtybLw1-13KK96S6aRmq0f6g75-mdptyNfEXqrEvFniiGtciYYEufgKeHRkmMHuA15O5wfxD9tArohmjMNuucQS84I6sI_wgKY.rl7g6Kc8GKutTmt6w4bX9SDzNMLu8mhZw3Ica5r5fSk&dib_tag=se&keywords=large+language+model+from+scratch&qid=1742825121&sprefix=large+language+model+from+scratch%2Caps%2C338&sr=8-1)

* MyGPT2文件夹 - gpt2模型的代码
* pretrain_on_unlabeled_text.ipynb - 使用verdict.txt作为语料预训练我们自定的gpt2模型；
* load_openai_weights.ipynb - 加载OpenAI预训练的参数到我们的gpt2模型中并使用；
* instruction_finetuning.ipynb - 加载OpenAI预训练的参数，然后用instruction-data.json来instruct finetune我们的gpt2模型；最终结果是模型可以follow简单的instruction来处理一些文本工作；例子见instruction-data.json；
* gpt_download.py 书本作者提供的下载OpenAI官方参数的脚本；