# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin
# https://hydra.cc/docs/advanced/search_path/

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ASECNTSearchPathPlugin(SearchPathPlugin):  # noqa: D101
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:  # noqa: D102
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.append(provider="ascent-searchpath-plugin", path="pkg://ascent.configs")
