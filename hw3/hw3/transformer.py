    mult_head = False
    padding = window_size // 2
    rows = seq_len
    cols = seq_len + (2 * padding)

    if len(q.size()) == 3:
        mult_head = True
        k = k.unsqueeze(1)
        q = q.unsqueeze(1)

    num_heads = q.shape[1]


    q = q.reshape(batch_size * num_heads, seq_len, embed_dim)
    k = k.reshape(batch_size * num_heads, seq_len, embed_dim).permute(0, 2, 1)

    attention_weights = torch.matmul(q.unsqueeze(2),
                                     torch.nn.functional.pad(k, (padding, padding)).unfold(-1, window_size + 1,
                                                                                           1).transpose(1, 2)).squeeze(
        2)
    index_diff = torch.arange(cols).unsqueeze(0) - torch.arange(rows).unsqueeze(1)
    valid_indices = (index_diff >= 0) & (index_diff < window_size + 1)
    valid_indices = valid_indices.repeat(batch_size * num_heads, 1).view(batch_size * num_heads, seq_len, -1)
    attention = torch.full((batch_size * num_heads, rows, cols), fill_value=float('-inf'), device=q.device)
    attention[valid_indices] = attention_weights.flatten()

    attention = attention[:, :, padding:-padding]
    attention /= math.sqrt(embed_dim)
    attention = attention.view(batch_size, num_heads, seq_len, seq_len)

    if mult_head:
        attention = attention.squeeze(1)
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)
        padding_mask = (padding_mask * padding_mask.transpose(-1, -2)).bool()
        padding_mask = padding_mask.expand(-1, num_heads, -1, -1)
        attention  = torch.where(padding_mask, attention, torch.full_like(attention, float('-inf')))
    attention = F.softmax(attention, dim=-1).nan_to_num(0.0)

    values = torch.matmul(attention, v)
    print('xx')