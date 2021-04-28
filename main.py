import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

def splitter(longtext_file_path,words_per_split=200):
    # Splits the long text into smaller batches

    # Reads files and counts words
    file = open(longtext_file_path, encoding='UTF-8')
    data = file.read()
    words = data.split()
    totalindex = len(words)
    print("Words:", totalindex)

    # Calculate how many splits we are going to do
    totalfiles = int(totalindex / words_per_split)+1
    print("Split into", totalfiles,"parts")

    # Split into the batches
    list_of_batches=[]
    for i in range(totalfiles):
        batch = words[i * words_per_split:i * words_per_split + words_per_split]
        list_of_batches.append(batch)
    return list_of_batches


def ask_question(longtext_file_path,question,early_stop_conf,words_per_split=200):
    # Load the model
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Used to keep track of the best batch
    currentmax = (0,0,'')

    # Call the above function for splitting
    list_of_batches = splitter(longtext_file_path,words_per_split)
    for inx in range(len(list_of_batches)):
        # Make the batch into a string
        answer_text = " ".join(list_of_batches[inx])
        # Tokenize
        input_ids = tokenizer.encode(question, answer_text)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for token, id in zip(tokens, input_ids):
            if id == tokenizer.sep_token_id:
                pass
            if id == tokenizer.sep_token_id:
                pass
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)

        outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
                        token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                        return_dict=True)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        # Combine the tokens in the answer and print it out.
        answer = ' '.join(tokens[answer_start:answer_end+1])
        answer=answer.replace('#','')
        # What we use to choose the best batch (The batch that has the highest confidence on first word)
        print(f"{inx}   {round(torch.max(start_scores).item(),4)}   {answer}")

        # Early exit (if the model is confident enough it wont continue, change this with early_stop_conf)
        thismax=torch.max(start_scores)
        if thismax.item() >early_stop_conf:
            print("")
            print("Early stop")
            print(inx, "Confidence:", thismax.item(), "   ANSWER:", answer)
            exit()
        # If this batch was more confident than the earlier best batch, but not confident enough for early exit
        if thismax.item() > currentmax[0]:
            currentmax = (inx, thismax.item(), answer)
    return currentmax



longtext_file_path = 'longtext.txt'
question = 'when was the winter war?'
early_stop_conf = 8       # How confident do you want the model to be to quit early (roughly 1-10 scale where >8 correct most of time)
words_per_split = 150     # How many words per batch, BERT can only handle 250 tokens so on average around 200 words


ans=ask_question(longtext_file_path,question,early_stop_conf,words_per_split)
print("")
print("index (batch):",ans[0])
print("Confidence",round(int(ans[1],2)))
print("answer:",ans[2])
