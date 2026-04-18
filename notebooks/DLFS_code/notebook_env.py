"""노트북 실행 환경을 Colab과 로컬에서 공통으로 맞추는 도우미 함수 모음.

이 저장소의 노트북은 원래 로컬 Jupyter 환경을 기준으로 작성되어 있다.
Colab에서는 다음 문제가 자주 생긴다.

1. 저장소 전체가 런타임에 없어서 ``lincoln`` 패키지를 찾지 못한다.
2. 현재 작업 디렉토리가 노트북이 있는 폴더가 아니라 상대 경로가 깨진다.
3. 한글 폰트가 운영체제마다 달라 그래프 축/제목이 깨진다.

이 모듈은 위 문제를 한 번에 해결하기 위해 만들어졌다.
노트북 첫 셀에서 ``prepare_notebook_environment()``만 호출하면
대부분의 환경 차이를 숨기고 동일한 코드로 실습을 진행할 수 있다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_URL = "https://github.com/glasslego/2026-ml-study.git"
REPO_NAME = "2026-ml-study"
DLFS_SUBDIR = "notebooks/DLFS_code"


def running_in_colab() -> bool:
    """현재 노트북이 Google Colab 안에서 실행 중인지 판단한다."""

    return "google.colab" in sys.modules


def _discover_repo_root(start: Path) -> Path:
    """현재 위치에서 저장소 루트를 거슬러 올라가며 찾는다.

    Args:
        start: 탐색을 시작할 기준 경로.

    Returns:
        README.md와 lincoln 폴더를 함께 가진 저장소 루트.

    Raises:
        FileNotFoundError: 시작 경로의 상위 어디에서도 저장소 루트를 못 찾은 경우.
    """

    candidates = [start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if (candidate / "README.md").exists() and (candidate / "lincoln").exists():
            return candidate

    raise FileNotFoundError(
        "DLFS_code 저장소 루트를 찾지 못했습니다. "
        "노트북을 저장소 내부에서 열었는지 확인하세요."
    )


def _ensure_colab_repo(repo_url: str = REPO_URL, repo_name: str = REPO_NAME) -> Path:
    """Colab 런타임에 저장소가 없으면 clone하고, 있으면 재사용한다.

    상위 저장소(glasslego/2026-ml-study)를 clone한 뒤
    실제 DLFS_code 루트는 ``notebooks/DLFS_code`` 하위에 있으므로
    그 경로를 저장소 루트로 반환한다.
    """

    clone_root = Path("/content") / repo_name
    if not clone_root.exists():
        print(f"Colab 런타임에 {repo_name} 저장소가 없어 새로 clone합니다.")
        subprocess.run(
            ["git", "clone", repo_url, str(clone_root)],
            check=True,
        )
    else:
        print(f"Colab 런타임에 기존 {repo_name} 저장소를 재사용합니다.")

    repo_root = clone_root / DLFS_SUBDIR
    if not repo_root.exists():
        raise FileNotFoundError(
            f"클론된 저장소에서 DLFS_code 경로({repo_root})를 찾지 못했습니다. "
            "DLFS_SUBDIR 값 또는 원격 저장소 구조를 확인하세요."
        )
    return repo_root


def resolve_repo_root(
    repo_url: str = REPO_URL,
    repo_name: str = REPO_NAME,
) -> Path:
    """실행 환경에 맞는 저장소 루트를 찾아 반환한다."""

    if running_in_colab():
        return _ensure_colab_repo(repo_url=repo_url, repo_name=repo_name)

    return _discover_repo_root(Path.cwd())


def _install_nanum_on_colab(verbose: bool = True) -> None:
    """Colab 런타임에 NanumGothic 폰트가 없으면 apt-get으로 설치한다.

    Colab 기본 이미지에는 한글 폰트가 없어서
    matplotlib가 DejaVu Sans로 fallback되고 그래프의 한글이 □ 로 깨진다.
    ``fonts-nanum`` 패키지를 설치하면 ``/usr/share/fonts/truetype/nanum/``
    아래에 NanumGothic 계열 ttf가 배치된다.
    """

    nanum_dir = Path("/usr/share/fonts/truetype/nanum")
    if nanum_dir.exists() and any(nanum_dir.glob("Nanum*.ttf")):
        return  # 이미 설치됨

    if verbose:
        print("Colab 런타임에 NanumGothic이 없어 apt-get으로 설치합니다...")
    try:
        subprocess.run(
            ["apt-get", "-qq", "install", "-y", "fonts-nanum"],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        if verbose:
            print(f"NanumGothic 설치 실패: {exc}. 그래프에서 한글이 깨질 수 있습니다.")


def configure_matplotlib_font(verbose: bool = True) -> Optional[str]:
    """사용 가능한 한글 폰트를 찾아 matplotlib 기본 폰트로 등록한다.

    Colab에서는 NanumGothic 패키지를 자동 설치하고, 설치 후에는
    ``font_manager.addfont``로 즉시 폰트를 등록해서
    커널 재시작 없이 바로 한글을 렌더링할 수 있게 한다.

    Args:
        verbose: 설정 결과를 출력할지 여부.

    Returns:
        선택된 폰트 이름. 적합한 폰트를 못 찾으면 ``None``.
    """

    try:
        import matplotlib
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("matplotlib가 아직 설치되지 않아 폰트 설정을 건너뜁니다.")
        return None

    # Colab이면 NanumGothic 설치 + matplotlib에 즉시 등록 (캐시 재빌드 없이)
    if running_in_colab():
        _install_nanum_on_colab(verbose=verbose)
        nanum_candidates = [
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ]
        for ttf in nanum_candidates:
            if ttf.exists():
                # addfont는 현재 런타임에서 새 폰트를 즉시 사용 가능하게 한다.
                font_manager.fontManager.addfont(str(ttf))

    candidates = [
        "NanumGothic",
        "AppleGothic",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    selected_font = next((font for font in candidates if font in available_fonts), None)
    if selected_font:
        plt.rc("font", family=selected_font)
        matplotlib.rcParams["axes.unicode_minus"] = False
        if verbose:
            print(f"matplotlib 기본 한글 폰트를 {selected_font}로 설정했습니다.")
        return selected_font

    if verbose:
        print("적절한 한글 폰트를 찾지 못해 matplotlib 기본 폰트를 유지합니다.")
    return None


def ensure_mnist_pickle(verbose: bool = True) -> Path:
    """현재 작업 디렉토리에 `mnist.pkl`이 없으면 자동으로 준비한다.

    일부 노트북은 `lincoln.utils.mnist.load()`를 바로 호출한다.
    이 함수는 현재 작업 디렉토리에 `mnist.pkl`이 이미 있다고 가정하므로,
    Colab 첫 실행에서는 사용자가 `mnist.init()`를 수동 실행해야 하는 불편이 있었다.
    초심자도 바로 실습을 시작할 수 있도록 데이터 준비를 공통 환경 단계로 끌어올린다.
    """

    mnist_path = Path.cwd() / "mnist.pkl"
    if mnist_path.exists():
        if verbose:
            print(f"기존 MNIST 캐시를 재사용합니다: {mnist_path}")
        return mnist_path

    from lincoln.utils import mnist

    if verbose:
        print(f"MNIST 캐시가 없어 새로 준비합니다: {mnist_path}")
    mnist.init()
    return mnist_path


def prepare_notebook_environment(
    notebook_dir: str,
    needs_lincoln: bool = False,
    ensure_mnist: bool = False,
    repo_url: str = REPO_URL,
    repo_name: str = REPO_NAME,
) -> Path:
    """노트북 실행에 필요한 공통 환경 구성을 끝낸다.

    이 함수가 수행하는 일은 다음과 같다.

    1. Colab이면 저장소 전체를 ``/content/DLFS_code`` 아래로 clone한다.
    2. 로컬이면 현재 위치에서 저장소 루트를 자동으로 찾는다.
    3. 노트북이 속한 폴더로 작업 디렉토리를 이동한다.
    4. 필요하면 ``lincoln`` 패키지가 import되도록 경로를 추가한다.
    5. 가능한 한글 폰트를 matplotlib에 등록한다.

    Args:
        notebook_dir: 저장소 루트를 기준으로 한 노트북 폴더 경로.
        needs_lincoln: ``lincoln`` 패키지를 import하는 노트북인지 여부.
        ensure_mnist: 현재 노트북이 `mnist.pkl`을 바로 필요로 하는지 여부.
        repo_url: Colab에서 clone할 원격 저장소 URL.
        repo_name: Colab에서 사용할 로컬 저장소 폴더 이름.

    Returns:
        탐지된 저장소 루트 경로.
    """

    repo_root = resolve_repo_root(repo_url=repo_url, repo_name=repo_name)
    target_dir = repo_root / notebook_dir

    if not target_dir.exists():
        raise FileNotFoundError(
            f"노트북 폴더 {target_dir} 를 찾지 못했습니다. notebook_dir 값을 확인하세요."
        )

    if str(repo_root) not in sys.path:
        # 저장소 루트를 import 경로에 넣으면 notebook_env 같은 루트 유틸리티를 불러올 수 있다.
        sys.path.insert(0, str(repo_root))

    lincoln_root = repo_root / "lincoln"
    if needs_lincoln and str(lincoln_root) not in sys.path:
        # lincoln은 별도 패키지 설치 없이 소스 디렉토리를 직접 import하도록 구성되어 있다.
        sys.path.insert(0, str(lincoln_root))

    os.chdir(target_dir)
    configure_matplotlib_font(verbose=True)

    print(f"저장소 루트: {repo_root}")
    print(f"현재 작업 디렉토리: {target_dir}")
    if needs_lincoln:
        print(f"lincoln import 경로 추가: {lincoln_root}")
    if ensure_mnist:
        ensure_mnist_pickle(verbose=True)

    return repo_root
