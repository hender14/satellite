# 簡易ｶﾙﾏﾝﾌｨﾙﾀｸﾗｽ
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_estimate):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_estimate

    def update(self, measurement):
        # 予測ｽﾃｯﾌﾟ
        prediction = self.estimate

        # 更新ｽﾃｯﾌﾟ
        kalman_gain = self.process_noise / (self.process_noise + self.measurement_noise)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.process_noise = (1 - kalman_gain) * self.process_noise

        return self.estimate