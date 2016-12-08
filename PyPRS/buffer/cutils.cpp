#include <iostream>
#include <limits>
#include <array>
#include <Python.h>
#pragma comment(lib, "C:\\Anaconda3\\libs\\python34.lib")
using namespace std;

bool dominating(int *p1, int *p2){
	float maxDiff = -std::numeric_limits<float>::infinity();
	float minDiff = std::numeric_limits<float>::infinity();
	int size = sizeof(p1) / sizeof(p1[0]);
	for (int i = 0; i<size; i++){
		int diff = p1[i] - p2[i];
		if (diff > maxDiff) maxDiff = diff;
		if (diff < minDiff) minDiff = diff;
	}
	bool flag = maxDiff <= 0 && minDiff < 0;
	return flag;
}

/* Exported function */
static PyObject * pyDominating(PyObject *self, PyObject *args)
{
    /* Define variables */
    int *p1;
	int *p2;
    bool flag;
    PyObject *obj;

    /* Parse Python args to C args */
    if (!PyArg_ParseTuple(args, "ii", &p1, &p2))
		return NULL;

    /* Call C native function */
    flag = dominating(p1, p2);

    /* Convert C variable to Python */
    obj = Py_BuildValue("i", flag);

    /* Return (converted)Python Object */
    return obj;
}

/* Method list */
static PyMethodDef CutilsMethods[] = {
/*	
	PyModuleDef_HEAD_INIT,
	"dominating",
	"Determine whether p1 dominates p2",
	-1,
	_dominating
*/	
    {"dominating", pyDominating, METH_VARARGS, "Determine whether p1 dominates p2"},
//	{ NULL, NULL, 0, NULL }
};

/* Module */
static struct PyModuleDef cutils = {
   PyModuleDef_HEAD_INIT,
   "cutils",   /* name of module */
   "", /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   CutilsMethods
};

/* Initialization function */
PyMODINIT_FUNC 
PyInit_cutils(void)
{
	return PyModule_Create(&cutils);
}