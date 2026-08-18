# SPDX-License-Identifier: MIT

from importlib.resources import files


def test_report_resources_are_packaged() -> None:
    report = files("skala.benchmark.report")
    expected = (
        report / "templates" / "index.html",
        report / "assets" / "report-base.css",
        report / "assets" / "report.css",
        report / "assets" / "report.js",
        report / "assets" / "vendor" / "d3.min.js",
        report / "assets" / "vendor" / "katex.min.js",
        report / "examples" / "prose.example.yaml",
    )

    assert all(resource.is_file() for resource in expected)
