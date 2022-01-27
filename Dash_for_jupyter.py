import re
import time
import random
import wfdb
import os
import pandas as pd
import dash
import math

from dash.dependencies import Input, Output, State
from dash import html, callback_context, dcc
from jupyter_dash import JupyterDash

import dash_bootstrap_components as dbc

# import dash_table
from dash import dash_table
import plotly.express as px


DEBUG = True

NUM_ATYPES = 13
DEFAULT_FIG_MODE = "layout"
SUB_GRAPH_NAMES = {
    "x": "i/mV",
    "x10": "v4/mV",
    "x11": "v5/mV",
    "x12": "v6/mV",
    "x2": "ii/mV",
    "x3": "iii/mV",
    "x4": "avr/mV",
    "x5": "avl/mV",
    "x6": "avf/mV",
    "x7": "v1/mV",
    "x8": "v2/mV",
    "x9": "v3/mV",
}
ANNOTATION_COLORMAP = px.colors.qualitative.Light24
ANNOTATION_TYPES = [
    "P",
    "Q",
    "R",
    "S",
    "T",
    "PR interval",
    "PR segment",
    "QRS complex",
    "ST segment",
    "QT interval",
    "OTHER",
]
DEFAULT_ATYPE = ANNOTATION_TYPES[2]

COLUMNS = [
    "Type",
    "Subplot",
    # "X0",
    # "Y0",
    # "X1",
    # "Y1",
    "Dist,sec",
    "Dist,mv",
]


def debug_print(*args):
    if DEBUG:
        print(*args)


def coord_to_tab_column(coord):
    return coord.upper()


def time_passed(start=0):
    return round(time.mktime(time.localtime())) - start


def format_float(f):
    return "%.2f" % (float(f),)


def shape_to_table_row(sh, type_dict):
    dist_x = 0
    dist_y = 0
    if sh["x1"]:
        dist_x = abs(float(sh["x1"]) - float(sh["x0"]))
        dist_y = abs(float(sh["y1"]) - float(sh["y0"]))
    sub_graph_name = SUB_GRAPH_NAMES[sh["xref"]]
    return {
        "Type": type_dict[sh["line"]["color"]],
        "Subplot": sub_graph_name,
        "X0": format_float(sh["x0"]),
        "Y0": format_float(sh["y0"]),
        "X1": format_float(sh["x1"]),
        "Y1": format_float(sh["y1"]),
        "XREF": sh["xref"],
        "YREF": sh["yref"],
        "Dist,sec": format_float(dist_x),
        "Dist,mv": format_float(dist_y),
    }


def default_table_row():
    return {
        "Type": DEFAULT_ATYPE,
        "X0": format_float(10),
        "Y0": format_float(10),
        "X1": format_float(20),
        "Y1": format_float(20),
    }


def table_row_to_shape(tr, color_dict):
    return {
        "editable": True,
        "xref": tr["XREF"],
        "yref": tr["YREF"],
        "layer": "above",
        "opacity": 1,
        "line": {
            "color": color_dict[tr["Type"]],
            "width": 4,
            "dash": "solid",
        },
        "fillcolor": "rgba(0, 0, 0, 0)",
        "fillrule": "evenodd",
        "type": "rect",
        "x0": tr["X0"],
        "y0": tr["Y0"],
        "x1": tr["X1"],
        "y1": tr["Y1"],
    }


def shape_cmp(s0, s1):
    """ Compare two shapes """
    return (
        all(s0[k] == s1[k] for k in ("x0", "x1", "y0", "y1"))
        and s0["line"]["color"] == s1["line"]["color"]  # noqa: W503
    )


def shape_in(se):
    """ check if a shape is in list (done this way to use custom compare) """
    return lambda s: any(shape_cmp(s, s_) for s_ in se)


def index_of_shape(shapes, shape):
    for i, shapes_item in enumerate(shapes):
        if shape_cmp(shapes_item, shape):
            return i
    raise ValueError  # not found


def annotations_table_shape_resize(annotations_table_data, fig_data):
    """
    Extract the shape that was resized (its index) and store the resized
    coordinates.
    """
    # debug_print("fig_data", fig_data)
    # debug_print("table_data", annotations_table_data)
    for key, val in fig_data.items():
        shape_nb, coord = key.split(".")
        # shape_nb is for example 'shapes[2].x0': this extracts the number
        shape_nb = shape_nb.split(".")[0].split("[")[-1].split("]")[0]
        # this should correspond to the same row in the data table
        # we have to format the float here because this is exactly the entry in
        # the table
        annotations_table_data[int(shape_nb)][
            coord_to_tab_column(coord)
        ] = format_float(fig_data[key])
        # (no need to compute a time stamp, that is done for any change in the
        # table values, so will be done later)
    return annotations_table_data


def shape_data_remove_timestamp(shape):
    """
    go.Figure complains if we include the 'timestamp' key when updating the
    figure
    """
    new_shape = dict()
    for k in shape.keys() - set(["timestamp"]):
        new_shape[k] = shape[k]
    return new_shape


def make_callbacks(
    app, color_dict, type_dict, rand_ECG_fig, num2field2values, all_columns
):
    """output_long = [
        Output(key + " " + str(num), "value")
        for num in (2, 1)
        for key in all_columns
    ]
    output_long.extend(
        [
            Output("file_description_expert", "data"),
            Output("file_description_student", "data"),
            Output("Result", "value"),
        ]
    )
    output_long.extend([Output(key + " 1", "style") for key in all_columns])
    print(output_long)"""

    # callback for graph anatitaion and table with coordinates update
    @app.callback(
        [
            Output("annotations-table", "data"),
            Output("image_files", "data"),
            Output("graph", "figure"),
            Output("annotations-store", "data"),
        ],
        [
            Input("previous", "n_clicks"),
            Input("next", "n_clicks"),
            Input("graph", "relayoutData"),
            Input("annotation-type-dropdown", "value"),
        ],
        [
            State("annotations-table", "data"),
            State("image_files", "data"),
            State("annotations-store", "data"),
        ],
    )
    def modify_table_entries(
        previous_n_clicks,
        next_n_clicks,
        graph_relayoutData,
        annotation_type,
        annotations_table_data,
        image_files_data,
        annotations_store_data,
    ):

        cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
        cbcontext_value = [p["value"] for p in dash.callback_context.triggered][
            0
        ]

        dragmode = ""
        if cbcontext_value and type(cbcontext_value) is dict:
            if "dragmode" in cbcontext_value.keys():
                dragmode = cbcontext_value["dragmode"]
                # debug_print("dragmode :", dragmode)

        # analysis of previous/next buttons' clicks
        image_index_change = 0
        if cbcontext == "previous.n_clicks":
            image_index_change = -1

        if cbcontext == "next.n_clicks":
            image_index_change = 1

        image_files_data["current"] += image_index_change
        image_files_data["current"] %= len(image_files_data["files"])

        filename = image_files_data["files"][image_files_data["current"]]
        # debug_print("filename: ", filename)

        if image_index_change != 0:
            # image changed, update annotations_table_data with new data
            graph_relayoutData = {}
            annotations_table_data = []
            filename = image_files_data["files"][image_files_data["current"]]
            for sh in annotations_store_data[filename]["shapes"]:
                annotations_table_data.append(shape_to_table_row(sh, type_dict))
            # debug_print("filename: ", filename)

        # Get a figureto work with
        fig = rand_ECG_fig[filename]

        # "uirevision" key let save actions like zoom in/out, pan etc.
        # till the next graph will be chosen
        fig.update_layout(uirevision=str(filename))

        if cbcontext_value:
            if (dragmode != "") or (cbcontext_value in (ANNOTATION_TYPES)):
                if (dragmode == "drawline") or (
                    cbcontext_value in (ANNOTATION_TYPES)
                ):
                    # print("annotation_type here", annotation_type)
                    fig.update_layout(
                        dragmode="drawline",
                        newshape_line_color=color_dict[annotation_type],
                    )
                else:
                    fig.update_layout(dragmode=str(dragmode))
        # add line to the graph and table with coordinates
        if graph_relayoutData:
            if "shapes" in graph_relayoutData.keys():
                # this means all the shapes have been passed to this function via
                # graph_relayoutData, so we store them
                annotations_table_data = [
                    shape_to_table_row(sh, type_dict)
                    for sh in graph_relayoutData["shapes"]
                ]
            elif re.match(
                r"shapes\[[0-9]+\].x0", list(graph_relayoutData.keys())[0]
            ):
                # this means a shape was updated (e.g., by clicking and dragging its
                # vertices), so we just update the specific shape
                annotations_table_data = annotations_table_shape_resize(
                    annotations_table_data, graph_relayoutData
                )
        if (
            annotations_table_data is not None
            and len(annotations_table_data) > 0
        ):
            # convert table rows to those understood by fig.u pdate_layout
            fig_shapes = [
                table_row_to_shape(sh, color_dict)
                for sh in annotations_table_data
            ]
            if fig_shapes:
                if "distance, mm/sec" in fig_shapes[0]:
                    del fig_shapes[0]["distance, mm/sec"]

            # find the shapes that are new
            new_shapes_i = []
            old_shapes_i = []
            for i, sh in enumerate(fig_shapes):
                if not shape_in(annotations_store_data[filename]["shapes"])(sh):
                    new_shapes_i.append(i)
                else:
                    old_shapes_i.append(i)
            # add timestamps to the new shapes
            for i in new_shapes_i:
                fig_shapes[i]["timestamp"] = time_passed(
                    annotations_store_data["starttime"]
                )

            # find the old shapes and look up their timestamps
            # debug_print("old_shapes_i:", old_shapes_i)
            for i in old_shapes_i:
                # debug_print("annotations_store: ", annotations_store_data)
                old_shape_i = index_of_shape(
                    annotations_store_data[filename]["shapes"],
                    fig_shapes[i],
                )
                fig_shapes[i]["timestamp"] = annotations_store_data[filename][
                    "shapes"
                ][old_shape_i]["timestamp"]
            shapes = fig_shapes
            # debug_print("shapes:", shapes)

            if "shapes" in graph_relayoutData.keys() and len(shapes) == 0:
                fig.__dict__["_layout_obj"]["shapes"] = ()
            elif "shapes" in graph_relayoutData.keys() and len(shapes) > 0:
                shapes_for_update = [
                    shape_data_remove_timestamp(sh) for sh in shapes
                ]
                # debug_print("shapes before update:", shapes)
                fig.__dict__["_layout_obj"]["shapes"] = shapes_for_update

            annotations_store_data[filename]["shapes"] = shapes

        return (
            annotations_table_data,
            image_files_data,
            fig,
            annotations_store_data,
        )

    # callback for description and comparison with expert's oppinion
    @app.callback(
        Output("modal", "is_open"),
        [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
        [State("modal", "is_open")],
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    @app.callback(
        [
            Output("ID 2", "value"),
            Output("Sex 2", "value"),
            Output("Age 2", "value"),
            Output("Rhythms 2", "value"),
            Output("Electric axis of the heart 2", "value"),
            Output("Conduction abnormalities 2", "value"),
            Output("Extrasystolies 2", "value"),
            Output("Hypertrophies 2", "value"),
            Output("Cardiac pacing 2", "value"),
            Output("Ischemia 2", "value"),
            Output("Non-specific repolarization abnormalities 2", "value"),
            Output("Other states 2", "value"),
            Output("ID 1", "value"),
            Output("Sex 1", "value"),
            Output("Age 1", "value"),
            Output("Rhythms 1", "value"),
            Output("Electric axis of the heart 1", "value"),
            Output("Conduction abnormalities 1", "value"),
            Output("Extrasystolies 1", "value"),
            Output("Hypertrophies 1", "value"),
            Output("Cardiac pacing 1", "value"),
            Output("Ischemia 1", "value"),
            Output("Non-specific repolarization abnormalities 1", "value"),
            Output("Other states 1", "value"),
            Output("file_description_expert", "data"),
            Output("file_description_student", "data"),
            Output("Result", "value"),
            Output("dropdowns_styles", "data"),
            Output("dropdowns_disabled", "data"),
            # Output("ID 1", "style"),
            # Output("Sex 1", "style"),
            # Output("Age 1", "style"),
            Output("Rhythms 1", "style"),
            Output("Electric axis of the heart 1", "style"),
            Output("Conduction abnormalities 1", "style"),
            Output("Extrasystolies 1", "style"),
            Output("Hypertrophies 1", "style"),
            Output("Cardiac pacing 1", "style"),
            Output("Ischemia 1", "style"),
            Output("Non-specific repolarization abnormalities 1", "style"),
            Output("Other states 1", "style"),
            Output("Rhythms 1", "disabled"),
            Output("Electric axis of the heart 1", "disabled"),
            Output("Conduction abnormalities 1", "disabled"),
            Output("Extrasystolies 1", "disabled"),
            Output("Hypertrophies 1", "disabled"),
            Output("Cardiac pacing 1", "disabled"),
            Output("Ischemia 1", "disabled"),
            Output("Non-specific repolarization abnormalities 1", "disabled"),
            Output("Other states 1", "disabled"),
        ],
        [
            Input("previous", "n_clicks"),
            Input("next", "n_clicks"),
            Input("btn-nclicks-1", "n_clicks"),
            Input("Rhythms 1", "value"),
            Input("Electric axis of the heart 1", "value"),
            Input("Conduction abnormalities 1", "value"),
            Input("Extrasystolies 1", "value"),
            Input("Hypertrophies 1", "value"),
            Input("Cardiac pacing 1", "value"),
            Input("Ischemia 1", "value"),
            Input("Non-specific repolarization abnormalities 1", "value"),
            Input("Other states 1", "value"),
        ],
        [
            State("image_files", "data"),
            State("file_description_expert", "data"),
            State("file_description_student", "data"),
            State("dropdowns_styles", "data"),
            State("dropdowns_disabled", "data"),
        ],
    )
    def displayClick(
        previous_clicks,
        next_clicks,
        btn1,
        rhythms_1,
        electric_axis_of_the_heart_1,
        conduction_abnormalities_1,
        extrasystolies_1,
        hypertrophies_1,
        cardiac_pacing_1,
        ischemia_1,
        non_specific_repolarization_abnormalities_1,
        other_states_1,
        image_files_data,
        file_description_exp,
        file_description_stud,
        dropdowns_styles,
        dropdowns_disabled,
    ):
        result_value = ""

        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        filename = image_files_data["files"][image_files_data["current"]]

        num_doc = int(
            filename[filename.find("/") + 1 : len(filename)]  # noqa: E203
        )
        field2values_exp = num2field2values[num_doc]

        # print("field2values_exp", field2values_exp)
        vals_from_dd = [
            rhythms_1,
            electric_axis_of_the_heart_1,
            conduction_abnormalities_1,
            extrasystolies_1,
            hypertrophies_1,
            cardiac_pacing_1,
            ischemia_1,
            non_specific_repolarization_abnormalities_1,
            other_states_1,
        ]

        # update studen't description based on selected values
        for idx, key in enumerate(
            list(file_description_stud[filename].keys())[3:]
        ):
            file_description_stud[filename].update({key: vals_from_dd[idx]})
        # update student's and expert's descriprions when previous/next button click
        if "btn-nclicks-1" in changed_id:
            for key in list(file_description_stud[filename].keys())[:3]:
                file_description_stud[filename].update(
                    {key: field2values_exp[key]}
                )

        if "btn-nclicks-1" in changed_id:
            for key in list(file_description_exp[filename].keys()):
                file_description_exp[filename].update(
                    {key: field2values_exp[key]}
                )
            # calculation of number of predicted correctly fields
            right_answ_count = []
            if (
                file_description_stud[filename]
                and file_description_exp[filename]
            ):
                for descr_dict in [
                    file_description_exp[filename],
                    file_description_stud[filename],
                ]:
                    for key, value in descr_dict.items():
                        if descr_dict[key] is None or descr_dict[key] == "":
                            descr_dict[key] = []

                for key, value in file_description_stud[filename].items():
                    right_answ_count.append(
                        sorted(file_description_stud[filename][key])
                        == sorted(file_description_exp[filename][key])
                    )

            result_value = f"Your result is: {sum(right_answ_count)-3} out of 9"

            # highligting all incorrectly predicted fields
            for i in range(len(right_answ_count[3:])):
                if right_answ_count[i] == 0:
                    dropdowns_styles[filename][i] = {
                        "background-color": "#FFD9D9"
                    }
            dropdowns_disabled[filename] = [
                True for key in list(file_description_stud[filename].keys())[3:]
            ]
        # analysis of previous/next buttons' clicks
        image_index_change = 0
        if changed_id == "previous.n_clicks":
            image_index_change = -1

        if changed_id == "next.n_clicks":
            image_index_change = 1

        image_files_data["current"] += image_index_change
        image_files_data["current"] %= len(image_files_data["files"])

        filename = image_files_data["files"][image_files_data["current"]]
        # debug_print("filename: ", filename)

        if image_index_change != 0:
            # image changed, update annotations_table_data with new data
            filename = image_files_data["files"][image_files_data["current"]]

        # debug_print(file_description_exp[filename])
        # debug_print(file_description_stud[filename])

        # creation whole return tuple
        return_ = ()
        for dict_descr in (file_description_exp, file_description_stud):
            return_ += tuple(
                dict_descr[filename][key]
                for key in list(dict_descr[filename].keys())
            )

        return_ += (
            file_description_exp,
            file_description_stud,
            result_value,
            dropdowns_styles,
            dropdowns_disabled,
        )
        return_ += tuple(dropdowns_styles[filename])
        return_ += tuple(dropdowns_disabled[filename])

        return return_

    # set the download url to the contents of the annotations-store (so they can be
    # downloaded from the browser's memory)

    app.clientside_callback(
        """
    function(the_store_data_1,annotation_dict)
    {
        let the_store_data =  Object.assign({}, the_store_data_1, {annotation_dict});
        let s = JSON.stringify(the_store_data);
        let b = new Blob([s],{type: 'text/plain'});
        let url = URL.createObjectURL(b);
        return url;
    }
    """,
        Output("download", "href"),
        [
            Input("file_description_student", "data"),
            Input("annotations-store", "data"),
        ],  #
    )

    # click on download link via button
    app.clientside_callback(
        """
    function(download_button_n_clicks)
    {
        let download_a=document.getElementById("download");
        download_a.click();
        return '';
    }
    """,
        Output("dummy", "children"),
        [Input("download-button", "n_clicks")],
    )

    # TODO comment the dbc link
    # we use a callback to toggle the collapse on small screens
    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open


def values_db_to_labels(values_db):
    r"""
    >>> values_db_to_labels(["c", "a\n", "a\nb", 1, float("nan")])
    ['1', 'a', 'b', 'c']

    ["c", "a", "a\nb"] ->
    [["c"], ["a"], ["a", "b"]] ->  # lists
    ["c", "a", "a", "b"] ->  # strings
    ["a", "b", "c"]
    """
    lists = (
        str(value).rstrip("\n").split("\n")
        for value in values_db
        if not (isinstance(value, float) and math.isnan(value))
    )
    strings = (string for list_ in lists for string in list_)
    return sorted(set(strings))


class Dash:
    def _init_num2field2values(self):
        ludb_description = pd.read_csv("ludb.csv")

        for pre in (
            "Non-specific repolarization abnormalities",
            "Electric axis of the heart",
        ):
            ludb_description[pre] = ludb_description[pre].str.replace(
                f"{pre}: ", ""
            )

        ludb_description = ludb_description.rename(
            columns={
                key: key.replace(" ", "_") for key in ludb_description.columns
            }
        )  # new method

        self.columns_df = list(ludb_description.columns)

        self.num2field2values = {
            row["ID"]: {
                key: values_db_to_labels([value]) for key, value in row.items()
            }
            for _, row in ludb_description.iterrows()
        }

    def __init__(self, folder_path):
        self._init_num2field2values()

        self.list_dictionries = []
        self.name2labels = {}
        self.name2labels_ids = {}

        for col_name in self.columns_df:
            labels = sorted(
                set(
                    value
                    for _num, field2values in self.num2field2values.items()
                    for value in field2values[col_name]
                )
            )
            # *values_db_to_labels(self.ludb_description[col_name].unique()),
            self.list_dictionries.append(labels)
            if col_name not in ("ID", "Age", "Sex"):
                self.name2labels[col_name] = labels
            else:
                self.name2labels_ids[col_name] = labels

        # folder_path = "C:/Users/fedyu/Desktop/UoM/Job/ISYS90069/lobachevsky-university-electrocardiography-database-1.0.1/lobachevsky-university-electrocardiography-database-1.0.1"

        # file_list_ = os.listdir(folder_path)
        record_list = pd.read_csv(
            os.path.join(folder_path, "RECORDS")
        ).values.ravel()
        # randomly select 5 rows from lobachevsky university electrocardiography database
        # random.seed(41)
        run_num = range(1, 20) if DEBUG else random.sample(range(200), 20)
        list_of_5_ECG = [record_list[i] for i in run_num]

        pids = list_of_5_ECG  # record_list[0]

        self.rand_ECG_fig = {}
        for pid in pids:
            record = wfdb.rdrecord(
                folder_path + "/" + str(pid)
            )  # os.path.join(folder_path_new,str(pid)))
            fig = wfdb.plot_wfdb_pl(
                record=record,
                time_units="seconds",
                height=1800,  # figsize=(1100, 1800),
                return_fig=True,
            )  # fig_test  #figsize=(1000,1800),

            self.rand_ECG_fig[pid] = fig

        with open("assets/Howto.md", "r") as f:
            # Using .read rather than .readlines because dcc.Markdown
            # joins list of strings with newline characters
            self.howto = f.read()

    def display(self):

        # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app = JupyterDash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "assets/image_annotation_style.css",
            ],
        )

        # prepare bijective type<->color mapping
        typ_col_pairs = [
            (t, ANNOTATION_COLORMAP[n % len(ANNOTATION_COLORMAP)])
            for n, t in enumerate(ANNOTATION_TYPES)
        ]
        # types to colors
        color_dict = {}
        # colors to types
        type_dict = {}
        for typ, col in typ_col_pairs:
            color_dict[typ] = col
            type_dict[col] = typ

        make_callbacks(
            app,
            color_dict,
            type_dict,
            self.rand_ECG_fig,
            self.num2field2values,
            list(self.name2labels_ids.keys()) + list(self.name2labels.keys()),
        )

        app.layout = self._make_layout(
            app, newshape_line_color=color_dict[DEFAULT_ATYPE]
        )

        app.run_server(mode="inline")

    def _make_stores(
        self,
        fig,
        filelist,
        file_description_expert,
        file_description_student,
        dropdowns_styles,
        dropdowns_disabled,
    ):
        return [
            dcc.Store(id="graph-copy", data=fig),
            dcc.Store(
                id="annotations-store",
                data=dict(
                    **{filename: {"shapes": []} for filename in filelist},
                    **{"starttime": time_passed()},
                ),
            ),
            dcc.Store(
                id="image_files",
                data={"files": filelist, "current": 0},
            ),
            # add new data
            dcc.Store(
                id="file_description_expert",
                data=file_description_expert,
            ),
            dcc.Store(
                id="file_description_student",
                data=file_description_student,
            ),
            dcc.Store(
                id="dropdowns_styles",
                data=dropdowns_styles,
            ),
            dcc.Store(
                id="dropdowns_disabled",
                data=dropdowns_disabled,
            ),
        ]

    def _make_buttons(self):
        # Buttons
        button_gh = dbc.Button(
            "Learn more",
            id="howto-open",
            outline=True,
            color="secondary",
            # Turn off lowercase transformation for class .button in stylesheet
            style={"textTransform": "none"},
        )

        button_howto = dbc.Button(
            "View Code on github",
            outline=True,
            color="primary",
            href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-annotation",
            id="gh-link",
            style={"text-transform": "none"},
        )
        return button_gh, button_howto

    def _make_rows_coordinates_of_annotations(
        self,
        fig,
        filelist,
        file_description_expert,
        file_description_student,
        dropdowns_styles,
        dropdowns_disabled,
    ):
        row_header = dbc.Row(dbc.Col(html.H3("Coordinates of annotations")))
        row_data = dbc.Row(
            dbc.Col(
                [
                    dash_table.DataTable(
                        id="annotations-table",
                        columns=[
                            dict(
                                name=n,
                                id=n,
                                presentation=(
                                    "dropdown" if n == "Type" else "input"
                                ),
                            )
                            for n in COLUMNS
                        ],
                        editable=True,
                        style_data={"height": 40},
                        style_cell={
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "maxWidth": 0,
                        },
                        dropdown={
                            "Type": {
                                "options": [
                                    {"label": o, "value": o}
                                    for o in ANNOTATION_TYPES
                                ],
                                "clearable": False,
                            }
                        },
                        style_cell_conditional=[
                            {
                                "if": {"column_id": "Type"},
                                "textAlign": "left",
                            }
                        ],
                        fill_width=True,
                    ),
                ],
            ),
        )
        return row_header, row_data

    def _make_dropdown(self, name: str, is_expert: bool) -> dcc.Dropdown:
        name_spaced = name.replace("_", " ")
        rv = html.Span(
            [
                html.Label(name_spaced + ":"),
                dcc.Dropdown(
                    id=f"{name_spaced} {2 if is_expert else 1}",
                    options=[
                        {"label": t, "value": t}
                        for t in self.name2labels[name]
                        # for t in self.list_dictionries[3]
                    ],
                    clearable=False,
                    disabled=is_expert,
                    multi=True,
                ),
            ]
        )
        # print(rv)
        return rv

    def _make_col(self, name: str, is_expert: bool) -> dbc.Col:
        name_spaced = name.replace("_", " ")
        rv = dbc.Col(
            children=[
                html.Span(
                    [
                        html.Label(name_spaced + ":"),
                        dcc.Dropdown(
                            id=f"{name_spaced} {2 if is_expert else 1}",
                            options=[
                                {"label": t, "value": t}
                                for t in self.name2labels_ids[name]
                                # for t in self.list_dictionries[3]
                            ],
                            clearable=False,
                            disabled=True,
                            multi=True,
                        ),
                    ]
                )
            ]
        )
        # print(rv)
        return rv

    def _make_layout(self, app, newshape_line_color):
        filelist = list(self.rand_ECG_fig.keys())

        file_description_expert = {}
        file_description_student = {}
        dropdowns_styles = {}
        dropdowns_disabled = {}

        for file in filelist:
            file_description_expert[file] = {}
            dropdowns_styles[file] = [{}] * 9
            dropdowns_disabled[file] = [False] * 9
            for col_name in self.columns_df:
                file_description_expert[file][col_name] = ""
            file_description_student[file] = {}
            for col_name in self.columns_df:
                file_description_student[file][col_name] = ""

        fig = self.rand_ECG_fig[filelist[0]]

        # Buttons
        button_gh, button_howto = self._make_buttons()

        # Modal
        modal_overlay = dbc.Modal(
            [
                dbc.ModalBody(
                    html.Div([dcc.Markdown(self.howto, id="howto-md")])
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="howto-close",
                        className="howto-bn",
                    )
                ),
            ],
            id="modal",
            size="lg",
            style={"font-size": "small"},
        )

        # Cards
        image_annotation_card = dbc.Card(
            id="imagebox",
            children=[
                dbc.CardHeader(html.H2("Annotation area")),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            id="graph",
                            figure=fig,
                            config={
                                "modeBarButtonsToAdd": [
                                    "drawline",
                                    "eraseshape",
                                ]
                            },  # drawrect
                            # scrolling  for the grahp
                            style={"overflow": "scroll", "height": "400px"},
                        )
                    ]
                ),
                dbc.CardFooter(
                    [
                        dcc.Markdown(
                            "To annotate the above ECG graph, select an appropriate label on the right and then draw a "
                            "line with your cursor between two points of the graph you wish to annotate.\n\n"
                            "**Choose a different ECG to annotate**:"
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Previous image",
                                    id="previous",
                                    outline=True,
                                ),
                                dbc.Button(
                                    "Next image", id="next", outline=True
                                ),
                            ],
                            size="lg",
                            style={"width": "100%"},
                        ),
                    ]
                ),
            ],
        )

        annotated_data_card = dbc.Card(
            [
                dbc.CardHeader(html.H2("Annotated data")),
                dbc.CardBody(
                    [
                        *self._make_rows_coordinates_of_annotations(
                            fig,
                            filelist,
                            file_description_expert,
                            file_description_student,
                            dropdowns_styles,
                            dropdowns_disabled,
                        ),
                        dbc.Row(
                            dbc.Col(
                                [
                                    html.H3("Create new annotation for"),
                                    dcc.Dropdown(
                                        id="annotation-type-dropdown",
                                        options=[
                                            {"label": t, "value": t}
                                            for t in ANNOTATION_TYPES
                                        ],
                                        value=DEFAULT_ATYPE,
                                        clearable=False,
                                    ),
                                ],
                                align="center",
                            )
                        ),
                        dbc.Row(
                            dbc.Col(
                                [
                                    html.Br(),
                                    html.H2("Description"),
                                ]
                            )
                        ),
                        dbc.Row(
                            [
                                dbc.Col([html.H3("Your description")]),
                                dbc.Col([html.H3("Expert description")]),
                            ]
                        ),
                        dbc.Row(
                            [
                                self._make_col(name, is_expert)
                                for is_expert in (False, True)
                                for name in self.name2labels_ids
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    children=[
                                        self._make_dropdown(name, is_expert)
                                        for name in self.name2labels
                                    ]
                                )
                                for is_expert in (False, True)
                            ]
                        ),
                        dbc.Row(
                            dbc.Col(
                                [
                                    html.P(),
                                    html.Label("By pressing this button:"),
                                    html.Br(),
                                    html.Button(
                                        "Submit",
                                        id="btn-nclicks-1",
                                        n_clicks=0,
                                    ),
                                    html.P(),
                                    html.Label("Result:"),
                                    html.Br(),
                                    dcc.Textarea(
                                        id="Result",
                                        value=" ",
                                        style={"width": "50%", "height": 30},
                                        disabled=True,
                                    ),
                                ]
                            )
                        ),
                    ]
                ),
                dbc.CardFooter(
                    [
                        html.Div(
                            [
                                # We use this pattern because we want to be able to download the
                                # annotations by clicking on a button
                                html.A(
                                    id="download",
                                    download="annotations.json",
                                    # make invisble, we just want it to click on it
                                    style={"display": "none"},
                                ),
                                dbc.Button(
                                    "Download annotations",
                                    id="download-button",
                                    outline=True,
                                ),
                                html.Div(id="dummy", style={"display": "none"}),
                                dbc.Tooltip(
                                    "You can download the annotated data in a .json format by clicking this button",
                                    target="download-button",
                                ),
                            ],
                        )
                    ]
                ),
            ],
        )

        # Navbar
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.A(
                                    html.Img(
                                        src=app.get_asset_url(
                                            "dash-logo-new.png"
                                        ),
                                        height="30px",
                                    ),
                                    href="https://plot.ly",
                                )
                            ),
                            dbc.Col(dbc.NavbarBrand("Image Annotation App")),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        dbc.Col(
                            [
                                dbc.NavbarToggler(id="navbar-toggler"),
                                dbc.Collapse(
                                    dbc.Nav(
                                        [
                                            dbc.NavItem(button_howto),
                                            dbc.NavItem(button_gh),
                                        ],
                                        className="ml-auto",
                                        navbar=True,
                                    ),
                                    id="navbar-collapse",
                                    navbar=True,
                                ),
                                modal_overlay,
                            ]
                        ),
                        align="center",
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-5",
        )

        return html.Div(
            [
                *self._make_stores(
                    fig,
                    filelist,
                    file_description_expert,
                    file_description_student,
                    dropdowns_styles,
                    dropdowns_disabled,
                ),
                navbar,
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(image_annotation_card, md=7),
                                dbc.Col(annotated_data_card, md=5),
                            ],
                        ),
                    ],
                    fluid=True,
                ),
            ]
        )
