from bffile._java_stuff import start_jvm


def test_setup_java() -> None:
    start_jvm()
