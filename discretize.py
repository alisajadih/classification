class discretize:
    def __init__(self, x):
        x['Stage'] = x['Stage'].apply(self.calc_stage)

    def calc_stage(self, value):
        lower_value = value.lower()
        if lower_value == 'won':
            return 2
        if lower_value == 'lost':
            return 0
        return 1
