# tf.gather
import tensorflow as tf
from tensor2tensor.layers import common_layers


def fast_tpu_gather(params, indices, name=None):
  """Fast gather implementation for models running on TPU.

  This function use one_hot and batch matmul to do gather, which is faster
  than gather_nd on TPU. For params that have dtype of int32 (sequences to
  gather from), batch_gather is used to keep accuracy.

  Args:
    params: A tensor from which to gather values.
      [batch_size, original_size, ...]
    indices: A tensor used as the index to gather values.
      [batch_size, selected_size].
    name: A string, name of the operation (optional).

  Returns:
    gather_result: A tensor that has the same rank as params.
      [batch_size, selected_size, ...]
  """
  with tf.name_scope(name):
    dtype = params.dtype

    def _gather(params, indices):
      """Fast gather using one_hot and batch matmul."""
      if dtype != tf.float32:
        params = tf.to_float(params)
      shape = common_layers.shape_list(params)
      indices_shape = common_layers.shape_list(indices)
      ndims = params.shape.ndims
      # Adjust the shape of params to match one-hot indices, which is the
      # requirement of Batch MatMul.
      if ndims == 2:
        params = tf.expand_dims(params, axis=-1)
      if ndims > 3:
        params = tf.reshape(params, [shape[0], shape[1], -1])
      gather_result = tf.matmul(
          tf.one_hot(indices, shape[1], dtype=params.dtype), params)
      if ndims == 2:
        gather_result = tf.squeeze(gather_result, axis=-1)
      if ndims > 3:
        shape[1] = indices_shape[1]
        gather_result = tf.reshape(gather_result, shape)
      if dtype != tf.float32:
        gather_result = tf.cast(gather_result, dtype)
      return gather_result

    # If the dtype is int, use the gather instead of one_hot matmul to avoid
    # precision loss. The max int value can be represented by bfloat16 in MXU is
    # 256, which is smaller than the possible id values. Encoding/decoding can
    # potentially used to make it work, but the benenfit is small right now.
    if dtype.is_integer:
      gather_result = tf.batch_gather(params, indices)
    else:
      gather_result = _gather(params, indices)

    return gather_result



# 3 * 10
x = tf.range(30)
x = tf.reshape(x, [3, 10])


idx = [[1], [1], [1]]
print(idx.shape)

res = fast_tpu_gather(x, idx)

with tf.Session() as sess:

    print(sess.run(x))
    print(sess.run(res))