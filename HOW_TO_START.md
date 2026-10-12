# Настройка SSH-ключа для GitHub на Windows (без GitHub Desktop)

## 1. Генерация ключа

1. Открой **Git Bash** (Пуск → начни вводить `Git Bash` → открой).
2. Вставь команду (замени email на свой от GitHub) и нажми **Enter**:

```bash
ssh-keygen -t ed25519 -C "твойemail@example.com"
```

3. Три раза нажми **Enter** (путь по умолчанию, пароль оставляем пустым).

## 2. Запуск агента и добавление ключа

В том же окне Git Bash выполни по очереди:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 3. Копирование ключа в буфер обмена

```bash
clip < ~/.ssh/id_ed25519.pub
```

## 4. Добавление ключа на GitHub

1. Открой: https://github.com/settings/keys
2. Нажми **New SSH key**.
3. В поле **Title** укажи, например, `My Windows PC`.
4. В поле **Key** нажми `Ctrl+V`.
5. Нажми **Add SSH key**.

## 5. Проверка подключения (опционально)

```bash
ssh -T git@github.com
```

При первом запуске спросит `Are you sure you want to continue connecting?` — введи `yes` и Enter.

Успех выглядит так:

```text
Hi твой_никнейм! You've successfully authenticated...
```

## 6. Клонирование репозитория

1. Открой страницу нужного репозитория на GitHub.
2. Нажми зелёную кнопку **Code** → вкладка **SSH** → скопируй ссылку (начинается с `git@github.com:...`).
3. В Git Bash перейди в нужную папку и клонируй:

```bash
cd ~/Desktop
git clone git@github.com:username/repo.git
```

Готово — репозиторий скачан.
