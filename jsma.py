import tensorflow as tf
import numpy as np


def jsma_impl_loop(sess,
                   x_value,
                   y_value,
                   model,
                   xi,
                   yi,
                   gamma=.1,
                   eps=1.0,
                   clip_min=0.0,
                   clip_max=1.0,
                   min_proba=0.0,
                   increase=True):

    epochs = gamma
    yi = tf.cast(tf.argmax(yi, axis=-1)[0], tf.int32)
    y_value = np.argmax(y_value, axis=-1)[0]
    if isinstance(epochs, float):
        tmp = np.prod(np.shape(x_value)) * epochs
        epochs = int(np.floor(tmp))

    def _body(x_adv, pixel_mask):
        ybar = model(x_adv)

        y_target = tf.slice(ybar, [0, yi], [-1, 1])
        dy_dx, = tf.gradients(ybar, x_adv)

        dt_dx, = tf.gradients(y_target, x_adv)
        do_dx = tf.subtract(dy_dx, dt_dx)
        score = tf.multiply(dt_dx, tf.abs(do_dx))

        cond = tf.logical_and(dt_dx >= 0, do_dx <= 0)
        domain = tf.logical_and(pixel_mask, cond)
        not_empty = tf.reduce_any(domain)

        # ensure that domain is not empty
        domain, score = tf.cond(not_empty, lambda: (domain, score),
                                lambda: (pixel_mask, dt_dx - do_dx))

        ind = tf.where(domain)
        score = tf.gather_nd(score, ind)

        p = tf.argmax(score, axis=0)
        p = tf.gather(ind, p)
        p = tf.expand_dims(p, axis=0)
        p = tf.to_int32(p)
        dx = tf.scatter_nd(p, [eps], tf.shape(x_adv), name='dx')

        if increase is False:
            dx = -1 * dx

        x_adv = tf.stop_gradient(x_adv + dx)
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        pixel_mask = tf.less(x_adv, clip_max)

        return x_adv, pixel_mask

    pixel_mask = tf.less(xi, clip_max)
    ybar = model(xi)
    x_adv_n, pixel_mask_new = _body(xi, pixel_mask)
    x_adv_n_np = x_value

    for i in range(epochs):
        # print(i)
        ybar_np = sess.run(ybar, feed_dict={xi: x_adv_n_np})
        # print(ybar_np)
        x_adv_n_np, pixel_mask_np = sess.run(
            [x_adv_n, pixel_mask_new], feed_dict={
                xi: x_adv_n_np,
                yi: y_value
            })
        # print(np.sum(pixel_mask_np))

    return x_adv_n_np
