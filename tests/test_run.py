import pytest
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from pathlib import Path

@pytest.fixture
def config_loader():
    return OmegaConfigLoader(conf_source=str(Path.cwd()))

@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="ft_medi_01",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
        env="local"
    )

class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()