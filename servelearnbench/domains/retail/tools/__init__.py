from .cancel_pending_order import CancelPendingOrder
from .exchange_delivered_order_items import ExchangeDeliveredOrderItems
from .find_user_id_by_name_zip import FindUserIdByNameZip
from .get_order_details import GetOrderDetails
from .get_product_details import GetProductDetails
from .get_user_details import GetUserDetails
from .place_order import PlaceOrder
from .read_docs import ReadDocs
from .return_delivered_order_items import ReturnDeliveredOrderItems

ALL_TOOLS = [
    ReadDocs,
    FindUserIdByNameZip,
    GetUserDetails,
    GetOrderDetails,
    GetProductDetails,
    CancelPendingOrder,
    ReturnDeliveredOrderItems,
    ExchangeDeliveredOrderItems,
    PlaceOrder,
]
