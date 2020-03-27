import functools
import glob
import multiprocessing
import tensorflow.compat.v1 as tf
from multiprocessing import Pool

flags = tf.flags

flags.DEFINE_string(
    'example_path',
    '/zfs1/hdaqing/saz31/dataset/example_v0_s2/*',
    'The path for ppdb outputs.')

FLAGS = flags.FLAGS
feature_set = {
        'src_ids': tf.FixedLenFeature([150], tf.int64),
        'trg_ids': tf.FixedLenFeature([150], tf.int64),
}


def decode(record):
    features = tf.parse_single_example(record, feature_set)
    return features['src_ids'], features['trg_ids']


def check(file, cache):
    if file in cache:
        return
    try:
        d = tf.data.TFRecordDataset(file)
        d = d.map(decode)
        d = d.repeat(1)
        d = d.batch(500)
        iterator = d.make_one_shot_iterator()
        src_wds, trg_wds = iterator.get_next()
        with tf.Session() as sess:
            _, _ = sess.run([src_wds, trg_wds])

        del d
        del iterator
    except tf.errors.OutOfRangeError:
        pass
    except Exception as e:
        print(file)
        # print(e)
    cache.add(file)


if __name__ == '__main__':
    p = Pool(30)
    cnt_cpu = multiprocessing.cpu_count()
    print(cnt_cpu)
    cache = set()
    files = glob.glob(FLAGS.example_path)
    # fn = lambda file: check(file, cache)
    fn = functools.partial(check, cache=cache)

    p.map(fn, files)


