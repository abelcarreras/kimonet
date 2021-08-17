#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif


//static double FrequencyEvaluation(double Delta, double  Coefficients[], int m, double xms);
static double GetKFactor(double  *u_d, double  *u_a , double  *r, int size);
double DotProduct(double *v, double *u, int n);
static double* UnitVector(double  *vector, int n);

static PyObject* DipoleExtended(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* Dipole(PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring_1[] =
    "dipole_extended(r_vector, mu_1, mu_2, n=1, longitude=1, n_divisions=10)\n\n Dipole-dipole interaction extended";
static char function_docstring_2[] =
    "dipole(r_vector, mu_1, mu_2, n=1)\n\n Dipole-dipole interaction";


static PyMethodDef extension_funcs[] = {
    {"dipole_extended",  (PyCFunction)DipoleExtended, METH_VARARGS|METH_KEYWORDS, function_docstring_1},
    {"dipole",  (PyCFunction)Dipole, METH_VARARGS|METH_KEYWORDS, function_docstring_2},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "forster",
  "forster module",
  -1,
  extension_funcs,
  NULL,
  NULL,
  NULL,
  NULL,
};
#endif


static PyObject *moduleinit(void)
{
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("forster",
        extension_funcs, "forster module");
#endif

  return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initforster(void)
    {
        import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_forster(void)
    {
        import_array();
        return moduleinit();
    }

#endif


double DotProduct(double *v, double *u, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; i++)
        result += v[i]*u[i];
    return result;
}

static double* UnitVector(double  *vector, int n)
{
    //double  r[n];
    double  *r = malloc(n * sizeof(double));

    double dot = DotProduct(vector, vector, n);

    for (int i=0; i< n; i++){
        r[i] = vector[i] / sqrt(dot);
    }

    return r;
}


static double GetKFactor(double  *Mu1, double  *Mu2 , double  *r, int size){

    double *n1 = UnitVector(Mu1, size);
    double *n2 = UnitVector(Mu2, size);
    double *e = UnitVector(r, size);
    double output;

    output =  DotProduct(n1, n2, size) - 3 *  DotProduct(e, n1, size) * DotProduct(e, n2, size) ;

    free(n1);
    free(n2);
    free(e);

    return output;

}


static PyObject* Dipole(PyObject* self, PyObject *arg, PyObject *keywords)
{

    double RefractiveIndex = 1;

    //  Interface with Python
    PyObject *r_vector_obj, *mu_1_obj, *mu_2_obj;
    static char *kwlist[] = {"r_vector", "mu_1", "mu_2", "n", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOO|d", kwlist, &mu_1_obj, &mu_2_obj, &r_vector_obj, &RefractiveIndex))  return NULL;

    PyObject *r_vector_array = PyArray_FROM_OTF(r_vector_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *mu_1_array = PyArray_FROM_OTF(mu_1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *mu_2_array = PyArray_FROM_OTF(mu_2_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (r_vector_array == NULL || mu_1_array == NULL || mu_2_array == NULL ) {
        Py_XDECREF(r_vector_array);
        Py_XDECREF(mu_1_array);
        Py_XDECREF(mu_2_array);
        return NULL;
    }

    double *RVector = (double*)PyArray_DATA(r_vector_array);
    double *Mu1     = (double*)PyArray_DATA(mu_1_array);
    double *Mu2     = (double*)PyArray_DATA(mu_2_array);
    int    NumberOfData = (int)PyArray_DIM(r_vector_array, 0);


    //Create new numpy array for storing result

    double KFactor = GetKFactor(Mu1, Mu2, RVector, NumberOfData);
    double Distance = sqrt(DotProduct(RVector, RVector, NumberOfData));
    double UnitsFactor = 1.0/(4.0 * 3.141592654 * 0.005524906526621038);  // (eV * Angs)/e^2

    double Coupling = UnitsFactor * KFactor * DotProduct(Mu1, Mu2, NumberOfData) / (pow(RefractiveIndex,2) * pow(Distance,3));

    // Free python memory
    Py_DECREF(r_vector_array);
    Py_DECREF(mu_1_array);
    Py_DECREF(mu_2_array);

    //Returning Python array
    return Py_BuildValue("d", Coupling);
}



static PyObject* DipoleExtended (PyObject* self, PyObject *arg, PyObject *keywords)
{

    double RefractiveIndex = 1.0;
    double Longitude = 1.0;
    int NDivisions = 10;

    //  Interface with Python
    PyObject *r_vector_obj, *mu_1_obj, *mu_2_obj;
    static char *kwlist[] = {"r_vector", "mu_1", "mu_2", "n", "longitude", "n_divisions", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOO|ddi", kwlist, &r_vector_obj, &mu_1_obj, &mu_2_obj,
                                     &RefractiveIndex, &Longitude, &NDivisions))  return NULL;

    PyObject *r_vector_array = PyArray_FROM_OTF(r_vector_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *mu_1_array = PyArray_FROM_OTF(mu_1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *mu_2_array = PyArray_FROM_OTF(mu_2_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (r_vector_array == NULL || mu_1_array == NULL || mu_2_array == NULL ) {
        Py_XDECREF(r_vector_array);
        Py_XDECREF(mu_1_array);
        Py_XDECREF(mu_2_array);
        return NULL;
    }

    double *RVector = (double*)PyArray_DATA(r_vector_array);
    double *Mu1     = (double*)PyArray_DATA(mu_1_array);
    double *Mu2     = (double*)PyArray_DATA(mu_2_array);
    int    NumberOfData = (int)PyArray_DIM(r_vector_array, 0);

    // dipole part
    double *Mu1i = malloc(NumberOfData * sizeof(double));
    double *Mu2i = malloc(NumberOfData * sizeof(double));
    double *RVectori = malloc(NumberOfData * sizeof(double));

    double *UVec1 = UnitVector(Mu1, NumberOfData);
    double *UVec2 = UnitVector(Mu2, NumberOfData);


    for (int i=0; i<NumberOfData; i++) {
        Mu1i[i] = Mu1[i] / NDivisions;
        Mu2i[i] = Mu2[i] / NDivisions;
    }

    double KFactor, Distance;
    double UnitsFactor = 1.0/(4.0 * 3.141592654 * 0.005524906526621038);  // (eV * Angs)/e^2

    double Coupling = 0;
    double x, y;
    for (int i=0; i<NDivisions; i++){
        for (int j=0; j<NDivisions; j++){

            x = (double)i/(double)NDivisions - 0.5 + 0.5/(double)NDivisions;
            y = (double)j/(double)NDivisions - 0.5 + 0.5/(double)NDivisions;
            //printf("x: %f %f\n", x, y);

            for (int k=0; k<NumberOfData; k++){
                //printf("%f ", UVec1[k] * x * Longitude);
                RVectori[k] = RVector[k] + UVec1[k] * x * Longitude - UVec2[k] * y * Longitude;
            }
            //printf("\n");

            KFactor = GetKFactor(Mu1i, Mu2i, RVectori, NumberOfData);
            Distance = sqrt(DotProduct(RVectori, RVectori, NumberOfData));
            //printf("-> %i %i %f\n", i, j, Coupling);
            Coupling += UnitsFactor * KFactor * DotProduct(Mu1i, Mu2i, NumberOfData) / (pow(RefractiveIndex,2) * pow(Distance,3));
        }
    }

    // Free c memory
    free(UVec1);
    free(UVec2);
    free(Mu1i);
    free(Mu2i);
    free(RVectori);

    // Free python memory
    Py_DECREF(r_vector_array);
    Py_DECREF(mu_1_array);
    Py_DECREF(mu_2_array);

    //Returning Python array
    return Py_BuildValue("d", Coupling);
}
