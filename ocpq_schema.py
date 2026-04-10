from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal, Any, Annotated

# --- TYPES AND VARIABLES ---

class VariableEvent(BaseModel):
    Event: int

class VariableObject(BaseModel):
    Object: int

QueryVariable = Union[VariableEvent, VariableObject]

# --- VALUE FILTERS ---

class ValueFilterFloat(BaseModel):
    type: Literal["Float"]
    min: Optional[float] = None
    max: Optional[float] = None

class ValueFilterInteger(BaseModel):
    type: Literal["Integer"]
    min: Optional[int] = None
    max: Optional[int] = None

class ValueFilterBoolean(BaseModel):
    type: Literal["Boolean"]
    is_true: bool

class ValueFilterString(BaseModel):
    type: Literal["String"]
    is_in: List[str]

class ValueFilterTime(BaseModel):
    type: Literal["Time"]
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None

QueryValueFilter = Annotated[Union[
    ValueFilterFloat, ValueFilterInteger, ValueFilterBoolean, 
    ValueFilterString, ValueFilterTime
], Field(discriminator="type")]

class TimepointAlways(BaseModel):
    type: Literal["Always"]

class TimepointSometime(BaseModel):
    type: Literal["Sometime"]

class TimepointAtEvent(BaseModel):
    type: Literal["AtEvent"]
    event: int

QueryObjectValueFilterTimepoint = Annotated[Union[
    TimepointAlways, TimepointSometime, TimepointAtEvent
], Field(discriminator="type")]

# --- OCPQ FILTERS ---

class FilterO2E(BaseModel):
    type: Literal["O2E"]
    object: int
    event: int
    qualifier: Optional[str] = None
    filter_label: Optional[str] = None

class FilterO2O(BaseModel):
    type: Literal["O2O"]
    object: int
    other_object: int
    qualifier: Optional[str] = None
    filter_label: Optional[str] = None

class FilterTimeBetweenEvents(BaseModel):
    type: Literal["TimeBetweenEvents"]
    from_event: int
    to_event: int
    min_seconds: Optional[float] = None
    max_seconds: Optional[float] = None

class FilterNotEqual(BaseModel):
    type: Literal["NotEqual"]
    var_1: QueryVariable
    var_2: QueryVariable

class FilterEventAttributeValueFilter(BaseModel):
    type: Literal["EventAttributeValueFilter"]
    event: int
    attribute_name: str
    value_filter: QueryValueFilter

class FilterObjectAttributeValueFilter(BaseModel):
    type: Literal["ObjectAttributeValueFilter"]
    object: int
    attribute_name: str
    at_time: QueryObjectValueFilterTimepoint
    value_filter: QueryValueFilter

class FilterBasicFilterCEL(BaseModel):
    type: Literal["BasicFilterCEL"]
    cel: str

QueryFilter = Annotated[Union[
    FilterO2E, FilterO2O, FilterTimeBetweenEvents, FilterNotEqual, 
    FilterEventAttributeValueFilter, FilterObjectAttributeValueFilter, FilterBasicFilterCEL
], Field(discriminator="type")]

# --- OCPQ SIZE FILTERS ---

class SizeFilterNumChilds(BaseModel):
    type: Literal["NumChilds"]
    child_name: str
    min: Optional[int] = None
    max: Optional[int] = None

QuerySizeFilter = Annotated[Union[SizeFilterNumChilds], Field(discriminator="type")]

# --- OCPQ CONSTRAINTS ---

class ConstraintFilterWrap(BaseModel):
    type: Literal["Filter"]
    filter: QueryFilter

class ConstraintSizeFilterWrap(BaseModel):
    type: Literal["SizeFilter"]
    filter: QuerySizeFilter

class ConstraintSAT(BaseModel):
    type: Literal["SAT"]
    child_names: List[str]

class ConstraintANY(BaseModel):
    type: Literal["ANY"]
    child_names: List[str]

class ConstraintNOT(BaseModel):
    type: Literal["NOT"]
    child_names: List[str]

class ConstraintOR(BaseModel):
    type: Literal["OR"]
    child_names: List[str]

class ConstraintAND(BaseModel):
    type: Literal["AND"]
    child_names: List[str]

QueryConstraint = Annotated[Union[
    ConstraintFilterWrap, ConstraintSizeFilterWrap, ConstraintSAT, 
    ConstraintANY, ConstraintNOT, ConstraintOR, ConstraintAND
], Field(discriminator="type")]

# --- TREE NODES ---

class BindingBoxData(BaseModel):
    newEventVars: Dict[str, List[str]]
    newObjectVars: Dict[str, List[str]]
    filters: List[QueryFilter]
    sizeFilters: List[QuerySizeFilter]
    constraints: List[QueryConstraint]
    evVarLabels: Dict[str, str]
    obVarLabels: Dict[str, str]

class BindingBoxTreeNodeBox(BaseModel):
    Box: tuple[BindingBoxData, List[int]]

class BindingBoxTree(BaseModel):
    nodes: List[BindingBoxTreeNodeBox]
    edgeNames: List[tuple[tuple[int, int], str]]

OCPQ_JSON_SCHEMA = BindingBoxTree.model_json_schema()
