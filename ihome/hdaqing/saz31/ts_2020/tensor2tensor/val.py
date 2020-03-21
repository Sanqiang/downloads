import tensorflow as tf

# def _expand_to_beam_size(tensor, beam_size):
#   """Tiles a given tensor by beam_size.
#
#   Args:
#     tensor: tensor to tile [batch_size, ...]
#     beam_size: How much to tile the tensor by.
#
#   Returns:
#     Tiled tensor [batch_size, beam_size, ...]
#   """
#   tensor = tf.expand_dims(tensor, axis=1)
#   tile_dims = [1] * tensor.shape.ndims
#   tile_dims[1] = beam_size
#
#   return tf.tile(tensor, tile_dims)
#
# sess = tf.Session()
# tensor = tf.constant([[[1,2],[1,2],[1,2]], [[1,2],[1,2],[1,2]]])
# tensor2 = _expand_to_beam_size(tensor, 4)
# res = sess.run(tensor2)
# print(tensor2)
# print(tensor)
# print(res)

# def compute_batch_indices(batch_size, beam_size):
#   """Computes the i'th coordinate that contains the batch index for gathers.
#
#   Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
#   batch the beam item is in. This will create the i of the i,j coordinate
#   needed for the gather.
#
#   Args:
#     batch_size: Batch size
#     beam_size: Size of the beam.
#   Returns:
#     batch_pos: [batch_size, beam_size] tensor of ids
#   """
#   batch_pos = tf.range(batch_size * beam_size) // beam_size
#   batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
#   return batch_pos
#
# batch_size = 2
# beam_size = 4
# syntax_level = 3
# length = 5
# alive_seq = tf.constant([
#     [[[1,2,3,4,0],[1,2,3,4,0],[1,2,3,4,5]], [[1,2,3,4,1],[1,2,3,4,1],[1,2,3,4,5]], [[1,2,3,4,2],[1,2,3,4,2],[1,2,3,4,5]], [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]],
#     [[[1,2,3,4,500],[1,2,3,4,5],[1,2,3,4,5]], [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]], [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]], [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]]
# ])
# print(alive_seq)
#
# batch_pos = compute_batch_indices(batch_size, beam_size//2)
# topk_coordinates = tf.stack([batch_pos,
#                              tf.constant([[0, 1],
#                                           [0, 0]], tf.int32)], axis=2)
# topk_seq = tf.gather_nd(alive_seq, topk_coordinates)
# print(topk_seq)
#
# sess = tf.Session()
# topk_seq = sess.run(topk_seq)
# print(topk_seq)
# idx = tf.stack(
#       [0 for _ in range(3)] + [0], axis=0)
# print(idx)
# x = tf.constant(
#     [
#         [[1,2,3,0], [2,0,4,0], [1,2,3,0]],
#         [[1,2,3,0], [2,0,4,0], [0,0,0,0]],
#     ]
# )
# print(x)
# z = tf.equal(x, idx)
# z = tf.reduce_all(z, axis=-1)
# print(z)
# sess = tf.Session()
# z = sess.run(z)
# print(z)

import spacy

nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])

def _get_spacytree_templatefull(sent):

    def rec(node, cur_output, outputs):
        cur_output.append(node.dep_)
        outputs[node.i][1] = list(cur_output)

        if len(list(node.children)) == 0:
            cur_output.pop()
            return

        for child in list(node.children):
            # outputs[child.i][1].append(cur_output[-1])
            rec(child, cur_output, outputs)
        cur_output.pop()

    doc = nlp(sent)
    outputs = ['#UNASSIGNED#' for _ in range(len(doc))]
    for token in doc:
        outputs[token.i] = [token.text, []]

    for token in doc:
        if token.dep_ == "ROOT":
            rec(token, [], outputs)
    assert all([tag != '#UNASSIGNED#' for tag in outputs])

    # Merge different tokenizers
    sent_split = sent.split()
    if len(outputs) != len(sent_split):
        foutputs = []
        i, j = 0, 0
        while i < len(outputs) and j < len(sent_split):
            if outputs[i][0] == sent_split[j]:
                foutputs.append(outputs[i])
                i += 1
                j += 1
            elif sent_split[j].startswith(outputs[i][0]):
                shortest_syntax = ['#UNASSIGNED#'] * 100
                tmp = sent_split[j]
                while i < len(outputs) and tmp.startswith(outputs[i][0]):
                    if len(outputs[i][1]) < len(shortest_syntax):
                        shortest_syntax = outputs[i][1]
                    tmp = tmp[len(outputs[i][0]):]
                    i += 1
                assert shortest_syntax[0] != '#UNASSIGNED#'
                foutputs.append((sent_split[j], shortest_syntax))
                j += 1
            else:
                raise ValueError("Unexpected Token: %s:%s" % (outputs, sent))
        outputs = foutputs

    template_full = []
    for output in outputs:
        template_full.append('|'.join(output[1]))

    assert len(template_full) == len(sent_split)

    return ' '.join(template_full)

_get_spacytree_templatefull('he invited gabrielli and others for a meeting in rome .')