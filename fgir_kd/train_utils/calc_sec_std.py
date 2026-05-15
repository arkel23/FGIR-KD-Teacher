import torch
from einops import reduce
from torch.nn import functional as F


def calc_std(output, calc_method = None, k = 3):
    if isinstance(output, tuple) and len(output) == 7:
        output, _, _, _, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 4:
        output, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 3:
        output, _, _ = output
    elif isinstance(output, tuple) and len(output) == 2:
        output, _ = output

    # determine wether we want to calculate the sec_std using softmax or log
    if calc_method == 'softmax':
        output = F.softmax(output, dim=1)
    elif calc_method == 'log':
        output = F.softmax(output, dim=1)
        output = torch.log(output)

    #input tensor with shape batch size * class
    #output tensor with shape 0 * 1
    top_k_v, _ = torch.topk(output, k+1, dim=1) # take the top_k predictions
    top_k_v_sliced = top_k_v[:, 1:]
    std_predictions = torch.std(top_k_v_sliced, dim=1)  # apply standard deviation
    mean_einops = reduce(std_predictions,'b -> 1','mean')
    mean_einops_scalar = mean_einops.item()

    return mean_einops_scalar


def calc_ratios(output, calc_method = None, ssp_top_k=3):
    if isinstance(output, tuple) and len(output) == 7:
        output, _, _, _, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 4:
        output, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 3:
        output, _, _ = output
    elif isinstance(output, tuple) and len(output) == 2:
        output, _ = output

    # determine wether we want to calculate the sec_std using softmax or log
    if calc_method == 'softmax':
        output = F.softmax(output, dim=1)
    elif calc_method == 'log':
        output = F.softmax(output, dim=1)
        output = torch.log(output)
    
    # implement the ratios metric and the entropy
    sorted_output,_ = torch.sort(output, 1, descending=True) # sort the input predictions tensor
    sliced_output = sorted_output[:, 1:ssp_top_k+1] # take the top 5 predictions from the model
    std_predictions = torch.std(sliced_output, dim=1) # calculate the secondary soft probabilities for each batch
    
    # take the top 3 predictions
    top_pred = sorted_output[:, 0]
    second_pred = sorted_output[:, 1]
    third_pred = sorted_output[:, 2]

    # calculate the ratios
    ratio_1_std = top_pred / std_predictions
    ratio_1_2 = top_pred/second_pred
    ratio_1_3 = top_pred/third_pred
    ratio_2_3 = second_pred/third_pred

    # calculate the average ratios for the whole batch
    ratio_1_std_final = torch.sum(ratio_1_std)/len(ratio_1_std)
    ratio_1_2_final = torch.sum(ratio_1_2)/len(ratio_1_2)
    ratio_1_3_final = torch.sum(ratio_1_3)/len(ratio_1_3)
    ratio_2_3_final = torch.sum(ratio_2_3)/len(ratio_2_3)

    return ratio_1_std_final, ratio_1_2_final, ratio_1_3_final, ratio_2_3_final

def calc_entropy(output):
    # default will always be softmax to prevent error
    if isinstance(output, tuple) and len(output) == 7:
        output, _, _, _, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 4:
        output, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 3:
        output, _, _ = output
    elif isinstance(output, tuple) and len(output) == 2:
        output, _ = output

    output = F.softmax(output, dim=1)

    entropy_value_final = 0
    log_p = torch.log(output) # calculate log_p
    p_log_p = output * log_p # calculate p(x)*log(p(x))
    entropy_value = -torch.sum(p_log_p, dim=1) # calculate the entropy
    entropy_value_final = torch.sum(entropy_value)/len(entropy_value) # calculate entropy among batch

    return entropy_value_final


def store_teacher_logits(output, teacher_logits, top_k=3):
    if isinstance(output, tuple) and len(output) == 7:
        output, _, _, _, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 4:
        output, _, _, _ = output
    elif isinstance(output, tuple) and len(output) == 3:
        output, _, _ = output
    elif isinstance(output, tuple) and len(output) == 2:
        output, _ = output

    # implement the ratios metric and the entropy
    sorted_output,_ = torch.sort(output, 1, descending=True) # sort the input predictions tensor

    top_preds = sorted_output[:, :top_k].detach().cpu()

    # take the top 3 predictions
    # top_pred = sorted_output[:, 0].detach().cpu()
    # second_pred = sorted_output[:, 1].detach().cpu()

    if teacher_logits is not None:
        teacher_logits = torch.cat([teacher_logits, top_preds], dim=0)
        # teacher_logits['top_preds'] = torch.cat([teacher_logits['top_preds'], top_preds], dim=0)
        # teacher_logits['top_1'] = torch.cat([teacher_logits['top_1'], top_pred], dim=0)
        # teacher_logits['top_2'] = torch.cat([teacher_logits['top_2'], second_pred], dim=0)
    else:
        teacher_logits = top_preds
        # teacher_logits['top_preds'] = top_preds
        # teacher_logits['top_1'] = top_pred
        # teacher_logits['top_2'] = second_pred

    return teacher_logits


if __name__ == '__main__':
    dummy_tensor = torch.rand(2, 5)
    calc_std(dummy_tensor)
    calc_ratios(dummy_tensor)
    calc_entropy(dummy_tensor)