# flask-server

- 잇다의 ML 서버입니다.

## 서버 실행

```bash
# 가상 환경 설치
$ python3 -m venv venv
```

```bash
# 가상 환경 실행
$ source venv/bin/activate
```

```bash
# 패키지 설치
$ pip3 install -r requirements.txt
```

```bash
# 플라스크 앱 설행 (8080 port)
$ python3 app.py
```

```bash
# 의존성을 저장할 경우
$ pip3 freeze > requirements.txt
```

## 환경 변수 설정

이 레포지토리는 public이기 때문에 .env을 다룰 때 신중하게 사용해야 합니다.
.env 파일을 생성한 후 필요하신 부분을 바꿔 쓰면 됩니다.

```bash
# 루트 디렉터리에 .env.example을 복사해 .env 파일 생성
$ cp .env.example .env
```
