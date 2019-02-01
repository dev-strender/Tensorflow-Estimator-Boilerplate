import tensorflow as tf


class SimpleNet(object):
    def __init__(self):
        self.learning_rate = 1e-4

    def model(self, *args, **kwargs):
        data_format = 'channels_last'

        if data_format == 'channels_first':
            input_shape = [3, 32, 32]
        else:
            assert data_format == 'channels_last'
            input_shape = [32, 32, 3]

        l = tf.keras.layers
        max_pool = l.MaxPooling2D(
            (2, 2), (2, 2), padding='same', data_format=data_format)
        # The model consists of a sequential chain of layers, so tf.keras.Sequential
        # (a subclass of tf.keras.Model) makes for a compact description.
        return tf.keras.Sequential(
            [
                l.Reshape(
                    target_shape=input_shape,
                    input_shape=(32 * 32,)),
                l.Conv2D(
                    32,
                    5,
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu),
                max_pool,
                l.Conv2D(
                    64, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
                max_pool,
                l.Flatten(),
                l.Dense(1024, activation=tf.nn.relu),
                l.Dropout(0.4),
                l.Dense(10)
            ])

    def model_fn(self, features, labels, mode):
        """The model function for creating an Estimator."""
        model = self.model()
        image = features
        if isinstance(image, dict):
            image = features['image']

        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = model(image, training=False)
            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                })
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            logits = model(image, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))

            # Name tensors to be logged with LoggingTensorHook.
            tf.identity(self.learning_rate, 'learning_rate')
            tf.identity(loss, 'cross_entropy')
            tf.identity(accuracy[1], name='train_accuracy')

            # Save accuracy scalar to Tensorboard output.
            tf.summary.scalar('train_accuracy', accuracy[1])

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

        if mode == tf.estimator.ModeKeys.EVAL:
            logits = model(image, training=False)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy':
                        tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1)),
                })