import logging
from types import SimpleNamespace

from calibrate import Calibrate_NetCDF

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OxygenSensor SensorID="38">
  <SerialNumber>2935</SerialNumber>
  <CalibrationDate>{date}</CalibrationDate>
</OxygenSensor>
"""


def test_cal_date_xml_files_searches_only_each_date_dir(tmp_path):
    """Each cal date dir must map to its own XML file, not the first one
    found anywhere under the serial number directory."""
    sensor_dir = tmp_path / "2935"
    for dirname, date in (("2014_Jun", "10-Jun-14"), ("2022_Apr", "06-Apr-22")):
        (sensor_dir / dirname).mkdir(parents=True)
        (sensor_dir / dirname / "Oxygen_SBE43_2935.xml").write_text(_XML.format(date=date))

    stub = SimpleNamespace(logger=logging.getLogger(__name__))
    cal_dates = Calibrate_NetCDF._cal_date_xml_files(
        stub, str(sensor_dir), ["2022_Apr", "2014_Jun"], 2935
    )

    assert [d.year for d in cal_dates] == [2014, 2022]  # noqa: S101
    assert cal_dates[next(iter(cal_dates))].endswith("2014_Jun/Oxygen_SBE43_2935.xml")  # noqa: S101
    assert list(cal_dates.values())[1].endswith("2022_Apr/Oxygen_SBE43_2935.xml")  # noqa: S101
