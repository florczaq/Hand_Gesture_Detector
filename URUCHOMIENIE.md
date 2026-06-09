# Instrukcja uruchomienia

## Wymagania

- **Docker** — [docker.com](https://www.docker.com/products/docker-desktop/)  
  Na Windows/macOS zainstaluj **Docker Desktop** (backend WSL 2 na Windows).
- **Kamera internetowa** podłączona do komputera.
- *(Tylko Windows — skrypty z kamerą)* **VcXsrv** — [sourceforge.net/projects/vcxsrv](https://sourceforge.net/projects/vcxsrv/)

---

## Pierwsze uruchomienie na Windows — jednorazowa konfiguracja

### 1. Włącz uruchamianie skryptów PowerShell
Otwórz **PowerShell jako administrator** i wpisz:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 2. Uruchom VcXsrv (serwer okien)
1. Otwórz **XLaunch** (instaluje się razem z VcXsrv).
2. Wybierz **Multiple windows** → **Dalej**.
3. Wybierz **Start no client** → **Dalej**.
4. Zaznacz **Disable access control** → **Dalej** → **Zakończ**.
5. Pozostaw ikonę VcXsrv w zasobniku — musi działać w tle.

---

## Krok 1 — Zbierz dane gestów

Przed uruchomieniem otwórz `register_cords/register_cords.py` i ustaw zmienną `GESTURE_LABEL` na nazwę gestu, który chcesz nagrać.

| System | Komenda |
|---|---|
| Windows (PowerShell) | `cd register_cords` → `.\run.ps1` |
| Linux / macOS | `cd register_cords` → `bash run.sh` |

- Naciśnij `S`, aby zapisać aktualną pozycję dłoni.
- Naciśnij `Esc`, aby zakończyć.

Pliki CSV trafią do folderu `data/train/`.

---

## Krok 2 — Wytrenuj model

| System | Komenda |
|---|---|
| Windows (PowerShell) | `cd learn_model` → `.\run.ps1` |
| Linux / macOS | `cd learn_model` → `bash run.sh` |

Model zostanie zapisany w folderze `models/`.

---

## Krok 3 — Uruchom detekcję na żywo

| System | Komenda |
|---|---|
| Windows (PowerShell) | `cd detector` → `.\run.ps1` |
| Linux / macOS | `cd detector` → `bash run.sh` |

Otworzy się okno z podglądem kamery. Wykryty gest będzie wyświetlany na ekranie.

---

## Wymuszenie przebudowy obrazu Docker

Do każdego skryptu można dodać flagę `--build`:

```powershell
.\run.ps1 --build   # Windows
bash run.sh --build # Linux / macOS
```

---

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---|---|
| `cannot connect to display` (Windows) | Sprawdź, czy VcXsrv działa i czy zaznaczyłeś "Disable access control". |
| `cannot connect to display` (Linux) | Upewnij się, że skrypt uruchamiasz z poziomu sesji graficznej (nie przez SSH bez przekazywania X11). |
| `could not open video device` (Windows) | Kamera musi być dostępna jako `/dev/video0` w WSL2 — może wymagać przekazania USB przez [`usbipd-win`](https://github.com/dorssel/usbipd-win). |
| `could not open video device` (Linux) | Sprawdź, czy urządzenie `/dev/video0` istnieje i czy użytkownik należy do grupy `video`. |
| `docker: command not found` | Zainstaluj Docker Desktop (Windows/macOS) lub Docker Engine (Linux) i zrestartuj terminal. |
| Skrypt `.ps1` nie uruchamia się | Uruchom `Set-ExecutionPolicy` zgodnie z sekcją "Pierwsze uruchomienie". |
