import os
import random
import tensorflow as tf

flags = tf.flags

flags.DEFINE_string(
    'example_path',
    '/zfs1/hdaqing/saz31/dataset/example_v3_s3_l64/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    'output_path',
    '/zfs1/hdaqing/saz31/dataset/example_v3_s3_l64_shuffle/',
    'The path for ppdb outputs.')

flags.DEFINE_integer('num_shard', 128, 'number of shared of outputs.')

FLAGS = flags.FLAGS

if __name__ == '__main__':
    os.makedirs(FLAGS.output_path, exist_ok=True)

    rng = random.Random(1234)
    for shard_idx in range(FLAGS.num_shard):
        output_path = os.path.join(FLAGS.output_path, 'shard_%s.example' % shard_idx)
        if os.path.exists(output_path):
            continue

        writer = tf.python_io.TFRecordWriter(output_path)

        examples = []

        # Process cur files
        cnt = 0
        for file in os.listdir(FLAGS.example_path):
            if hash(file) % FLAGS.num_shard != shard_idx:
                continue
            tmp_examples = []
            cur_example_path = os.path.join(FLAGS.example_path, file)
            for example in tf.python_io.tf_record_iterator(cur_example_path):
                tmp_examples.append(example)

            rng.shuffle(tmp_examples)
            examples.extend(tmp_examples)
            cnt += 1

        rng.shuffle(examples)
        for example in examples:
            writer.write(example)
        writer.close()
        print('Done with file %s of size %s' % (output_path, cnt))
