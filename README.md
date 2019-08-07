# Distillation 을 활용한 Super-Resolution 연구의 base pytorch framework

# 기본 spec


## Training

학습은 크게 두 step 으로 진행된다.
1. step 1 에서는 teacher network 를 학습한다.
1. step 2 에서는 teacher network 의 정보를 distill 하여 student network 를 학습한다.

그 외, teacher network 로부터 distillation 을 받지 않았을 때의 성능을 측정하기 위해 step 0. 을 추가하였다.

Tips. 실험 방법이 크게 변하더라도 되도록이면 train.py 파일은 건드리지 않도록 한다. loss function 과 model 에 필요한 부분을 추가하고, config 파일을 통해 실험 setting 을 관리하면서 실험하도록 한다. train.py 파일들을 불가피하게 바꿔야 할 경우 이전 실험들과의 compatibility 를 신경쓰면서 파일을 변경하도록 한다. 만약 compatible 하게 유지할 수 없을 경우 train.py 파일을 새로 만들도록 한다. 이 때는 그 실험에 해당하는 step 0 ~ 2 를 모두 만들도록 한다.

## Configs
configs 폴더 내에는 실험 방법을 폴더 명으로 하는 yaml 파일들의 상위 폴더를 만들도록 한다. 각 폴더 내에는 base.yml 과 step0~2.yml 이 들어있다.

1. base.yml
base.yml 은 나머지 세 yml 파일에서 공통으로 쓰이는 configuration 을 담고있으며, 크게 data 에 관한 부분과 train directory 를 설정해놓는다. data의 train, valid, test 는 각 상황에서 사용될 데이터셋과 그 스펙을 명시하고있으며, list 형식을 지원하기때문에 여러 종류의 데이터셋을 함께 묶어서 dataset loader 에 전달할 수 있다. 예를 들면, data.test 에 주석해 놓은 부분을 풀면 Set5 와 Set14 를 함께 test 할 수 있다. 이는 train 과 valid 에서도 마찬가지이다.

1. step0~2.yml


## TODO
1.


