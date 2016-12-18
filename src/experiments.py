# This experiment is an averaged histogram given a data set of
#    single-valued points. It uses broadcast addition and also 
#    a "higher level" operation (foldl) to process the histogram shifts

import tensorflow as tf

# Performs one histogram with the given range, accumulating the counts
def one_histogram(accumulator, element):
    hist = tf.histogram_fixed_width(data, element, num_bins)
    return tf.add(accumulator, hist)

def average_histogram(data, num_bins, num_shifts):
    # Set up the range tensor
    data_min = tf.reduce_min(data)
    data_max = tf.reduce_max(data)
    data_range = [data_min, data_max]
    range_tensor_2d = tf.expand_dims(data_range, 0)

    # Set up the shift tensor
    half_bin = (data_max - data_min) / (num_bins * 2)
    shift = tf.linspace(-half_bin, half_bin, num_shifts)
    shift_tensor_2d = tf.expand_dims(shift, 1)

    # Create the range elements for the shifts using broadcast addition
    range_elements = tf.add(range_tensor_2d, shift_tensor_2d)

    # Perform the shifted histograms and divide by the total count
    initial_accumulator = tf.zeros([num_bins], tf.int32)
    sum_hist = tf.foldl(one_histogram, range_elements, initial_accumulator)
    total = tf.to_float(tf.reduce_sum(sum_hist))
    avg_hist = tf.div(tf.to_float(sum_hist), total)
    
    return avg_hist

# Run the session
# Set up the data - for testing, using normal distribution
normal = tf.random_normal([10000], 0.0, 100.0)
result = average_histogram(normal, 100, 10)
sess = tf.Session()
print(sess.run(result))
