import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class PreTrainDataset(Dataset):
    '''
    参考 https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-2.html#p174
    Input和Target的长度是一样的, Target是Input往左移动一位
    
    比如句子 "one two three four five six seven", 
    Input是["one", "two", "three", "four"], Target就是["two", "three", "four", "five"], 这样和LLM的输出结构吻合, 
    即和输入相同index的输出, 是model predict的下一个字;
    比如输入"three", LLM在attention mask后会把注意力放在"one", "two", 然后在对应的index输出下一个字"four";

    这样保证了在inference的时候, 无论输入长度是多少, 只取输入长度最后一个index的输出即是model predict的下一个字;
    '''
    def __init__(self, text, tokenizer, sample_len, stride) -> None:
        super().__init__()
        self.input_samples = []
        self.target_samples = []
        token_ids = tokenizer.encode(text)

        # stride = sample_len 防止sample间有重叠
        # 使用stride的话根据range的机制，末尾的一些token会被舍弃
        for i in range(0, len(token_ids) - sample_len, stride):
            self.input_samples.append(torch.tensor(token_ids[i:i+sample_len]))
            j = i + 1
            self.target_samples.append(torch.tensor(token_ids[j:j+sample_len]))

    def __len__(self) -> int:
        return len(self.input_samples)
    
    def __getitem__(self, index):
        return self.input_samples[index], self.target_samples[index]


def create_data_loader(text, batch_size=4, sample_len=256, 
                       stride=128, shuffle=True, drop_last=True, num_workers=0):
    '''
    create a pytorch DataLoader for model training
    '''
    tokeninzer = tiktoken.get_encoding("gpt2")
    dataset = PreTrainDataset(text, tokeninzer, sample_len, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )