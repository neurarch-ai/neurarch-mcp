var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __esm = (fn, res, err) => function __init() {
  if (err) throw err[0];
  try {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  } catch (e) {
    throw err = [e], e;
  }
};
var __commonJS = (cb, mod) => function __require() {
  try {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  } catch (e) {
    throw mod = 0, e;
  }
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// src/utils/paramEstimator.ts
var init_paramEstimator = __esm({
  "src/utils/paramEstimator.ts"() {
    "use strict";
  }
});

// ../Neurarch/node_modules/react/cjs/react.production.min.js
var require_react_production_min = __commonJS({
  "../Neurarch/node_modules/react/cjs/react.production.min.js"(exports) {
    "use strict";
    var l = /* @__PURE__ */ Symbol.for("react.element");
    var n = /* @__PURE__ */ Symbol.for("react.portal");
    var p = /* @__PURE__ */ Symbol.for("react.fragment");
    var q = /* @__PURE__ */ Symbol.for("react.strict_mode");
    var r = /* @__PURE__ */ Symbol.for("react.profiler");
    var t = /* @__PURE__ */ Symbol.for("react.provider");
    var u = /* @__PURE__ */ Symbol.for("react.context");
    var v = /* @__PURE__ */ Symbol.for("react.forward_ref");
    var w = /* @__PURE__ */ Symbol.for("react.suspense");
    var x = /* @__PURE__ */ Symbol.for("react.memo");
    var y = /* @__PURE__ */ Symbol.for("react.lazy");
    var z = Symbol.iterator;
    function A(a) {
      if (null === a || "object" !== typeof a) return null;
      a = z && a[z] || a["@@iterator"];
      return "function" === typeof a ? a : null;
    }
    var B = { isMounted: function() {
      return false;
    }, enqueueForceUpdate: function() {
    }, enqueueReplaceState: function() {
    }, enqueueSetState: function() {
    } };
    var C = Object.assign;
    var D = {};
    function E(a, b, e) {
      this.props = a;
      this.context = b;
      this.refs = D;
      this.updater = e || B;
    }
    E.prototype.isReactComponent = {};
    E.prototype.setState = function(a, b) {
      if ("object" !== typeof a && "function" !== typeof a && null != a) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
      this.updater.enqueueSetState(this, a, b, "setState");
    };
    E.prototype.forceUpdate = function(a) {
      this.updater.enqueueForceUpdate(this, a, "forceUpdate");
    };
    function F() {
    }
    F.prototype = E.prototype;
    function G(a, b, e) {
      this.props = a;
      this.context = b;
      this.refs = D;
      this.updater = e || B;
    }
    var H = G.prototype = new F();
    H.constructor = G;
    C(H, E.prototype);
    H.isPureReactComponent = true;
    var I = Array.isArray;
    var J = Object.prototype.hasOwnProperty;
    var K = { current: null };
    var L = { key: true, ref: true, __self: true, __source: true };
    function M(a, b, e) {
      var d, c = {}, k = null, h = null;
      if (null != b) for (d in void 0 !== b.ref && (h = b.ref), void 0 !== b.key && (k = "" + b.key), b) J.call(b, d) && !L.hasOwnProperty(d) && (c[d] = b[d]);
      var g = arguments.length - 2;
      if (1 === g) c.children = e;
      else if (1 < g) {
        for (var f = Array(g), m = 0; m < g; m++) f[m] = arguments[m + 2];
        c.children = f;
      }
      if (a && a.defaultProps) for (d in g = a.defaultProps, g) void 0 === c[d] && (c[d] = g[d]);
      return { $$typeof: l, type: a, key: k, ref: h, props: c, _owner: K.current };
    }
    function N(a, b) {
      return { $$typeof: l, type: a.type, key: b, ref: a.ref, props: a.props, _owner: a._owner };
    }
    function O(a) {
      return "object" === typeof a && null !== a && a.$$typeof === l;
    }
    function escape(a) {
      var b = { "=": "=0", ":": "=2" };
      return "$" + a.replace(/[=:]/g, function(a2) {
        return b[a2];
      });
    }
    var P = /\/+/g;
    function Q(a, b) {
      return "object" === typeof a && null !== a && null != a.key ? escape("" + a.key) : b.toString(36);
    }
    function R(a, b, e, d, c) {
      var k = typeof a;
      if ("undefined" === k || "boolean" === k) a = null;
      var h = false;
      if (null === a) h = true;
      else switch (k) {
        case "string":
        case "number":
          h = true;
          break;
        case "object":
          switch (a.$$typeof) {
            case l:
            case n:
              h = true;
          }
      }
      if (h) return h = a, c = c(h), a = "" === d ? "." + Q(h, 0) : d, I(c) ? (e = "", null != a && (e = a.replace(P, "$&/") + "/"), R(c, b, e, "", function(a2) {
        return a2;
      })) : null != c && (O(c) && (c = N(c, e + (!c.key || h && h.key === c.key ? "" : ("" + c.key).replace(P, "$&/") + "/") + a)), b.push(c)), 1;
      h = 0;
      d = "" === d ? "." : d + ":";
      if (I(a)) for (var g = 0; g < a.length; g++) {
        k = a[g];
        var f = d + Q(k, g);
        h += R(k, b, e, f, c);
      }
      else if (f = A(a), "function" === typeof f) for (a = f.call(a), g = 0; !(k = a.next()).done; ) k = k.value, f = d + Q(k, g++), h += R(k, b, e, f, c);
      else if ("object" === k) throw b = String(a), Error("Objects are not valid as a React child (found: " + ("[object Object]" === b ? "object with keys {" + Object.keys(a).join(", ") + "}" : b) + "). If you meant to render a collection of children, use an array instead.");
      return h;
    }
    function S(a, b, e) {
      if (null == a) return a;
      var d = [], c = 0;
      R(a, d, "", "", function(a2) {
        return b.call(e, a2, c++);
      });
      return d;
    }
    function T(a) {
      if (-1 === a._status) {
        var b = a._result;
        b = b();
        b.then(function(b2) {
          if (0 === a._status || -1 === a._status) a._status = 1, a._result = b2;
        }, function(b2) {
          if (0 === a._status || -1 === a._status) a._status = 2, a._result = b2;
        });
        -1 === a._status && (a._status = 0, a._result = b);
      }
      if (1 === a._status) return a._result.default;
      throw a._result;
    }
    var U = { current: null };
    var V = { transition: null };
    var W = { ReactCurrentDispatcher: U, ReactCurrentBatchConfig: V, ReactCurrentOwner: K };
    function X() {
      throw Error("act(...) is not supported in production builds of React.");
    }
    exports.Children = { map: S, forEach: function(a, b, e) {
      S(a, function() {
        b.apply(this, arguments);
      }, e);
    }, count: function(a) {
      var b = 0;
      S(a, function() {
        b++;
      });
      return b;
    }, toArray: function(a) {
      return S(a, function(a2) {
        return a2;
      }) || [];
    }, only: function(a) {
      if (!O(a)) throw Error("React.Children.only expected to receive a single React element child.");
      return a;
    } };
    exports.Component = E;
    exports.Fragment = p;
    exports.Profiler = r;
    exports.PureComponent = G;
    exports.StrictMode = q;
    exports.Suspense = w;
    exports.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = W;
    exports.act = X;
    exports.cloneElement = function(a, b, e) {
      if (null === a || void 0 === a) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + a + ".");
      var d = C({}, a.props), c = a.key, k = a.ref, h = a._owner;
      if (null != b) {
        void 0 !== b.ref && (k = b.ref, h = K.current);
        void 0 !== b.key && (c = "" + b.key);
        if (a.type && a.type.defaultProps) var g = a.type.defaultProps;
        for (f in b) J.call(b, f) && !L.hasOwnProperty(f) && (d[f] = void 0 === b[f] && void 0 !== g ? g[f] : b[f]);
      }
      var f = arguments.length - 2;
      if (1 === f) d.children = e;
      else if (1 < f) {
        g = Array(f);
        for (var m = 0; m < f; m++) g[m] = arguments[m + 2];
        d.children = g;
      }
      return { $$typeof: l, type: a.type, key: c, ref: k, props: d, _owner: h };
    };
    exports.createContext = function(a) {
      a = { $$typeof: u, _currentValue: a, _currentValue2: a, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null };
      a.Provider = { $$typeof: t, _context: a };
      return a.Consumer = a;
    };
    exports.createElement = M;
    exports.createFactory = function(a) {
      var b = M.bind(null, a);
      b.type = a;
      return b;
    };
    exports.createRef = function() {
      return { current: null };
    };
    exports.forwardRef = function(a) {
      return { $$typeof: v, render: a };
    };
    exports.isValidElement = O;
    exports.lazy = function(a) {
      return { $$typeof: y, _payload: { _status: -1, _result: a }, _init: T };
    };
    exports.memo = function(a, b) {
      return { $$typeof: x, type: a, compare: void 0 === b ? null : b };
    };
    exports.startTransition = function(a) {
      var b = V.transition;
      V.transition = {};
      try {
        a();
      } finally {
        V.transition = b;
      }
    };
    exports.unstable_act = X;
    exports.useCallback = function(a, b) {
      return U.current.useCallback(a, b);
    };
    exports.useContext = function(a) {
      return U.current.useContext(a);
    };
    exports.useDebugValue = function() {
    };
    exports.useDeferredValue = function(a) {
      return U.current.useDeferredValue(a);
    };
    exports.useEffect = function(a, b) {
      return U.current.useEffect(a, b);
    };
    exports.useId = function() {
      return U.current.useId();
    };
    exports.useImperativeHandle = function(a, b, e) {
      return U.current.useImperativeHandle(a, b, e);
    };
    exports.useInsertionEffect = function(a, b) {
      return U.current.useInsertionEffect(a, b);
    };
    exports.useLayoutEffect = function(a, b) {
      return U.current.useLayoutEffect(a, b);
    };
    exports.useMemo = function(a, b) {
      return U.current.useMemo(a, b);
    };
    exports.useReducer = function(a, b, e) {
      return U.current.useReducer(a, b, e);
    };
    exports.useRef = function(a) {
      return U.current.useRef(a);
    };
    exports.useState = function(a) {
      return U.current.useState(a);
    };
    exports.useSyncExternalStore = function(a, b, e) {
      return U.current.useSyncExternalStore(a, b, e);
    };
    exports.useTransition = function() {
      return U.current.useTransition();
    };
    exports.version = "18.3.1";
  }
});

// ../Neurarch/node_modules/react/cjs/react.development.js
var require_react_development = __commonJS({
  "../Neurarch/node_modules/react/cjs/react.development.js"(exports, module) {
    "use strict";
    if (process.env.NODE_ENV !== "production") {
      (function() {
        "use strict";
        if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ !== "undefined" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart === "function") {
          __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(new Error());
        }
        var ReactVersion = "18.3.1";
        var REACT_ELEMENT_TYPE = /* @__PURE__ */ Symbol.for("react.element");
        var REACT_PORTAL_TYPE = /* @__PURE__ */ Symbol.for("react.portal");
        var REACT_FRAGMENT_TYPE = /* @__PURE__ */ Symbol.for("react.fragment");
        var REACT_STRICT_MODE_TYPE = /* @__PURE__ */ Symbol.for("react.strict_mode");
        var REACT_PROFILER_TYPE = /* @__PURE__ */ Symbol.for("react.profiler");
        var REACT_PROVIDER_TYPE = /* @__PURE__ */ Symbol.for("react.provider");
        var REACT_CONTEXT_TYPE = /* @__PURE__ */ Symbol.for("react.context");
        var REACT_FORWARD_REF_TYPE = /* @__PURE__ */ Symbol.for("react.forward_ref");
        var REACT_SUSPENSE_TYPE = /* @__PURE__ */ Symbol.for("react.suspense");
        var REACT_SUSPENSE_LIST_TYPE = /* @__PURE__ */ Symbol.for("react.suspense_list");
        var REACT_MEMO_TYPE = /* @__PURE__ */ Symbol.for("react.memo");
        var REACT_LAZY_TYPE = /* @__PURE__ */ Symbol.for("react.lazy");
        var REACT_OFFSCREEN_TYPE = /* @__PURE__ */ Symbol.for("react.offscreen");
        var MAYBE_ITERATOR_SYMBOL = Symbol.iterator;
        var FAUX_ITERATOR_SYMBOL = "@@iterator";
        function getIteratorFn(maybeIterable) {
          if (maybeIterable === null || typeof maybeIterable !== "object") {
            return null;
          }
          var maybeIterator = MAYBE_ITERATOR_SYMBOL && maybeIterable[MAYBE_ITERATOR_SYMBOL] || maybeIterable[FAUX_ITERATOR_SYMBOL];
          if (typeof maybeIterator === "function") {
            return maybeIterator;
          }
          return null;
        }
        var ReactCurrentDispatcher = {
          /**
           * @internal
           * @type {ReactComponent}
           */
          current: null
        };
        var ReactCurrentBatchConfig = {
          transition: null
        };
        var ReactCurrentActQueue = {
          current: null,
          // Used to reproduce behavior of `batchedUpdates` in legacy mode.
          isBatchingLegacy: false,
          didScheduleLegacyUpdate: false
        };
        var ReactCurrentOwner = {
          /**
           * @internal
           * @type {ReactComponent}
           */
          current: null
        };
        var ReactDebugCurrentFrame = {};
        var currentExtraStackFrame = null;
        function setExtraStackFrame(stack) {
          {
            currentExtraStackFrame = stack;
          }
        }
        {
          ReactDebugCurrentFrame.setExtraStackFrame = function(stack) {
            {
              currentExtraStackFrame = stack;
            }
          };
          ReactDebugCurrentFrame.getCurrentStack = null;
          ReactDebugCurrentFrame.getStackAddendum = function() {
            var stack = "";
            if (currentExtraStackFrame) {
              stack += currentExtraStackFrame;
            }
            var impl = ReactDebugCurrentFrame.getCurrentStack;
            if (impl) {
              stack += impl() || "";
            }
            return stack;
          };
        }
        var enableScopeAPI = false;
        var enableCacheElement = false;
        var enableTransitionTracing = false;
        var enableLegacyHidden = false;
        var enableDebugTracing = false;
        var ReactSharedInternals = {
          ReactCurrentDispatcher,
          ReactCurrentBatchConfig,
          ReactCurrentOwner
        };
        {
          ReactSharedInternals.ReactDebugCurrentFrame = ReactDebugCurrentFrame;
          ReactSharedInternals.ReactCurrentActQueue = ReactCurrentActQueue;
        }
        function warn(format) {
          {
            {
              for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
                args[_key - 1] = arguments[_key];
              }
              printWarning("warn", format, args);
            }
          }
        }
        function error(format) {
          {
            {
              for (var _len2 = arguments.length, args = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
                args[_key2 - 1] = arguments[_key2];
              }
              printWarning("error", format, args);
            }
          }
        }
        function printWarning(level, format, args) {
          {
            var ReactDebugCurrentFrame2 = ReactSharedInternals.ReactDebugCurrentFrame;
            var stack = ReactDebugCurrentFrame2.getStackAddendum();
            if (stack !== "") {
              format += "%s";
              args = args.concat([stack]);
            }
            var argsWithFormat = args.map(function(item) {
              return String(item);
            });
            argsWithFormat.unshift("Warning: " + format);
            Function.prototype.apply.call(console[level], console, argsWithFormat);
          }
        }
        var didWarnStateUpdateForUnmountedComponent = {};
        function warnNoop(publicInstance, callerName) {
          {
            var _constructor = publicInstance.constructor;
            var componentName = _constructor && (_constructor.displayName || _constructor.name) || "ReactClass";
            var warningKey = componentName + "." + callerName;
            if (didWarnStateUpdateForUnmountedComponent[warningKey]) {
              return;
            }
            error("Can't call %s on a component that is not yet mounted. This is a no-op, but it might indicate a bug in your application. Instead, assign to `this.state` directly or define a `state = {};` class property with the desired state in the %s component.", callerName, componentName);
            didWarnStateUpdateForUnmountedComponent[warningKey] = true;
          }
        }
        var ReactNoopUpdateQueue = {
          /**
           * Checks whether or not this composite component is mounted.
           * @param {ReactClass} publicInstance The instance we want to test.
           * @return {boolean} True if mounted, false otherwise.
           * @protected
           * @final
           */
          isMounted: function(publicInstance) {
            return false;
          },
          /**
           * Forces an update. This should only be invoked when it is known with
           * certainty that we are **not** in a DOM transaction.
           *
           * You may want to call this when you know that some deeper aspect of the
           * component's state has changed but `setState` was not called.
           *
           * This will not invoke `shouldComponentUpdate`, but it will invoke
           * `componentWillUpdate` and `componentDidUpdate`.
           *
           * @param {ReactClass} publicInstance The instance that should rerender.
           * @param {?function} callback Called after component is updated.
           * @param {?string} callerName name of the calling function in the public API.
           * @internal
           */
          enqueueForceUpdate: function(publicInstance, callback, callerName) {
            warnNoop(publicInstance, "forceUpdate");
          },
          /**
           * Replaces all of the state. Always use this or `setState` to mutate state.
           * You should treat `this.state` as immutable.
           *
           * There is no guarantee that `this.state` will be immediately updated, so
           * accessing `this.state` after calling this method may return the old value.
           *
           * @param {ReactClass} publicInstance The instance that should rerender.
           * @param {object} completeState Next state.
           * @param {?function} callback Called after component is updated.
           * @param {?string} callerName name of the calling function in the public API.
           * @internal
           */
          enqueueReplaceState: function(publicInstance, completeState, callback, callerName) {
            warnNoop(publicInstance, "replaceState");
          },
          /**
           * Sets a subset of the state. This only exists because _pendingState is
           * internal. This provides a merging strategy that is not available to deep
           * properties which is confusing. TODO: Expose pendingState or don't use it
           * during the merge.
           *
           * @param {ReactClass} publicInstance The instance that should rerender.
           * @param {object} partialState Next partial state to be merged with state.
           * @param {?function} callback Called after component is updated.
           * @param {?string} Name of the calling function in the public API.
           * @internal
           */
          enqueueSetState: function(publicInstance, partialState, callback, callerName) {
            warnNoop(publicInstance, "setState");
          }
        };
        var assign = Object.assign;
        var emptyObject = {};
        {
          Object.freeze(emptyObject);
        }
        function Component(props, context, updater) {
          this.props = props;
          this.context = context;
          this.refs = emptyObject;
          this.updater = updater || ReactNoopUpdateQueue;
        }
        Component.prototype.isReactComponent = {};
        Component.prototype.setState = function(partialState, callback) {
          if (typeof partialState !== "object" && typeof partialState !== "function" && partialState != null) {
            throw new Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
          }
          this.updater.enqueueSetState(this, partialState, callback, "setState");
        };
        Component.prototype.forceUpdate = function(callback) {
          this.updater.enqueueForceUpdate(this, callback, "forceUpdate");
        };
        {
          var deprecatedAPIs = {
            isMounted: ["isMounted", "Instead, make sure to clean up subscriptions and pending requests in componentWillUnmount to prevent memory leaks."],
            replaceState: ["replaceState", "Refactor your code to use setState instead (see https://github.com/facebook/react/issues/3236)."]
          };
          var defineDeprecationWarning = function(methodName, info) {
            Object.defineProperty(Component.prototype, methodName, {
              get: function() {
                warn("%s(...) is deprecated in plain JavaScript React classes. %s", info[0], info[1]);
                return void 0;
              }
            });
          };
          for (var fnName in deprecatedAPIs) {
            if (deprecatedAPIs.hasOwnProperty(fnName)) {
              defineDeprecationWarning(fnName, deprecatedAPIs[fnName]);
            }
          }
        }
        function ComponentDummy() {
        }
        ComponentDummy.prototype = Component.prototype;
        function PureComponent(props, context, updater) {
          this.props = props;
          this.context = context;
          this.refs = emptyObject;
          this.updater = updater || ReactNoopUpdateQueue;
        }
        var pureComponentPrototype = PureComponent.prototype = new ComponentDummy();
        pureComponentPrototype.constructor = PureComponent;
        assign(pureComponentPrototype, Component.prototype);
        pureComponentPrototype.isPureReactComponent = true;
        function createRef() {
          var refObject = {
            current: null
          };
          {
            Object.seal(refObject);
          }
          return refObject;
        }
        var isArrayImpl = Array.isArray;
        function isArray(a) {
          return isArrayImpl(a);
        }
        function typeName(value) {
          {
            var hasToStringTag = typeof Symbol === "function" && Symbol.toStringTag;
            var type = hasToStringTag && value[Symbol.toStringTag] || value.constructor.name || "Object";
            return type;
          }
        }
        function willCoercionThrow(value) {
          {
            try {
              testStringCoercion(value);
              return false;
            } catch (e) {
              return true;
            }
          }
        }
        function testStringCoercion(value) {
          return "" + value;
        }
        function checkKeyStringCoercion(value) {
          {
            if (willCoercionThrow(value)) {
              error("The provided key is an unsupported type %s. This value must be coerced to a string before before using it here.", typeName(value));
              return testStringCoercion(value);
            }
          }
        }
        function getWrappedName(outerType, innerType, wrapperName) {
          var displayName = outerType.displayName;
          if (displayName) {
            return displayName;
          }
          var functionName = innerType.displayName || innerType.name || "";
          return functionName !== "" ? wrapperName + "(" + functionName + ")" : wrapperName;
        }
        function getContextName(type) {
          return type.displayName || "Context";
        }
        function getComponentNameFromType(type) {
          if (type == null) {
            return null;
          }
          {
            if (typeof type.tag === "number") {
              error("Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue.");
            }
          }
          if (typeof type === "function") {
            return type.displayName || type.name || null;
          }
          if (typeof type === "string") {
            return type;
          }
          switch (type) {
            case REACT_FRAGMENT_TYPE:
              return "Fragment";
            case REACT_PORTAL_TYPE:
              return "Portal";
            case REACT_PROFILER_TYPE:
              return "Profiler";
            case REACT_STRICT_MODE_TYPE:
              return "StrictMode";
            case REACT_SUSPENSE_TYPE:
              return "Suspense";
            case REACT_SUSPENSE_LIST_TYPE:
              return "SuspenseList";
          }
          if (typeof type === "object") {
            switch (type.$$typeof) {
              case REACT_CONTEXT_TYPE:
                var context = type;
                return getContextName(context) + ".Consumer";
              case REACT_PROVIDER_TYPE:
                var provider = type;
                return getContextName(provider._context) + ".Provider";
              case REACT_FORWARD_REF_TYPE:
                return getWrappedName(type, type.render, "ForwardRef");
              case REACT_MEMO_TYPE:
                var outerName = type.displayName || null;
                if (outerName !== null) {
                  return outerName;
                }
                return getComponentNameFromType(type.type) || "Memo";
              case REACT_LAZY_TYPE: {
                var lazyComponent = type;
                var payload = lazyComponent._payload;
                var init = lazyComponent._init;
                try {
                  return getComponentNameFromType(init(payload));
                } catch (x) {
                  return null;
                }
              }
            }
          }
          return null;
        }
        var hasOwnProperty = Object.prototype.hasOwnProperty;
        var RESERVED_PROPS = {
          key: true,
          ref: true,
          __self: true,
          __source: true
        };
        var specialPropKeyWarningShown, specialPropRefWarningShown, didWarnAboutStringRefs;
        {
          didWarnAboutStringRefs = {};
        }
        function hasValidRef(config) {
          {
            if (hasOwnProperty.call(config, "ref")) {
              var getter = Object.getOwnPropertyDescriptor(config, "ref").get;
              if (getter && getter.isReactWarning) {
                return false;
              }
            }
          }
          return config.ref !== void 0;
        }
        function hasValidKey(config) {
          {
            if (hasOwnProperty.call(config, "key")) {
              var getter = Object.getOwnPropertyDescriptor(config, "key").get;
              if (getter && getter.isReactWarning) {
                return false;
              }
            }
          }
          return config.key !== void 0;
        }
        function defineKeyPropWarningGetter(props, displayName) {
          var warnAboutAccessingKey = function() {
            {
              if (!specialPropKeyWarningShown) {
                specialPropKeyWarningShown = true;
                error("%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://reactjs.org/link/special-props)", displayName);
              }
            }
          };
          warnAboutAccessingKey.isReactWarning = true;
          Object.defineProperty(props, "key", {
            get: warnAboutAccessingKey,
            configurable: true
          });
        }
        function defineRefPropWarningGetter(props, displayName) {
          var warnAboutAccessingRef = function() {
            {
              if (!specialPropRefWarningShown) {
                specialPropRefWarningShown = true;
                error("%s: `ref` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://reactjs.org/link/special-props)", displayName);
              }
            }
          };
          warnAboutAccessingRef.isReactWarning = true;
          Object.defineProperty(props, "ref", {
            get: warnAboutAccessingRef,
            configurable: true
          });
        }
        function warnIfStringRefCannotBeAutoConverted(config) {
          {
            if (typeof config.ref === "string" && ReactCurrentOwner.current && config.__self && ReactCurrentOwner.current.stateNode !== config.__self) {
              var componentName = getComponentNameFromType(ReactCurrentOwner.current.type);
              if (!didWarnAboutStringRefs[componentName]) {
                error('Component "%s" contains the string ref "%s". Support for string refs will be removed in a future major release. This case cannot be automatically converted to an arrow function. We ask you to manually fix this case by using useRef() or createRef() instead. Learn more about using refs safely here: https://reactjs.org/link/strict-mode-string-ref', componentName, config.ref);
                didWarnAboutStringRefs[componentName] = true;
              }
            }
          }
        }
        var ReactElement = function(type, key, ref, self, source, owner, props) {
          var element = {
            // This tag allows us to uniquely identify this as a React Element
            $$typeof: REACT_ELEMENT_TYPE,
            // Built-in properties that belong on the element
            type,
            key,
            ref,
            props,
            // Record the component responsible for creating this element.
            _owner: owner
          };
          {
            element._store = {};
            Object.defineProperty(element._store, "validated", {
              configurable: false,
              enumerable: false,
              writable: true,
              value: false
            });
            Object.defineProperty(element, "_self", {
              configurable: false,
              enumerable: false,
              writable: false,
              value: self
            });
            Object.defineProperty(element, "_source", {
              configurable: false,
              enumerable: false,
              writable: false,
              value: source
            });
            if (Object.freeze) {
              Object.freeze(element.props);
              Object.freeze(element);
            }
          }
          return element;
        };
        function createElement(type, config, children) {
          var propName;
          var props = {};
          var key = null;
          var ref = null;
          var self = null;
          var source = null;
          if (config != null) {
            if (hasValidRef(config)) {
              ref = config.ref;
              {
                warnIfStringRefCannotBeAutoConverted(config);
              }
            }
            if (hasValidKey(config)) {
              {
                checkKeyStringCoercion(config.key);
              }
              key = "" + config.key;
            }
            self = config.__self === void 0 ? null : config.__self;
            source = config.__source === void 0 ? null : config.__source;
            for (propName in config) {
              if (hasOwnProperty.call(config, propName) && !RESERVED_PROPS.hasOwnProperty(propName)) {
                props[propName] = config[propName];
              }
            }
          }
          var childrenLength = arguments.length - 2;
          if (childrenLength === 1) {
            props.children = children;
          } else if (childrenLength > 1) {
            var childArray = Array(childrenLength);
            for (var i = 0; i < childrenLength; i++) {
              childArray[i] = arguments[i + 2];
            }
            {
              if (Object.freeze) {
                Object.freeze(childArray);
              }
            }
            props.children = childArray;
          }
          if (type && type.defaultProps) {
            var defaultProps = type.defaultProps;
            for (propName in defaultProps) {
              if (props[propName] === void 0) {
                props[propName] = defaultProps[propName];
              }
            }
          }
          {
            if (key || ref) {
              var displayName = typeof type === "function" ? type.displayName || type.name || "Unknown" : type;
              if (key) {
                defineKeyPropWarningGetter(props, displayName);
              }
              if (ref) {
                defineRefPropWarningGetter(props, displayName);
              }
            }
          }
          return ReactElement(type, key, ref, self, source, ReactCurrentOwner.current, props);
        }
        function cloneAndReplaceKey(oldElement, newKey) {
          var newElement = ReactElement(oldElement.type, newKey, oldElement.ref, oldElement._self, oldElement._source, oldElement._owner, oldElement.props);
          return newElement;
        }
        function cloneElement(element, config, children) {
          if (element === null || element === void 0) {
            throw new Error("React.cloneElement(...): The argument must be a React element, but you passed " + element + ".");
          }
          var propName;
          var props = assign({}, element.props);
          var key = element.key;
          var ref = element.ref;
          var self = element._self;
          var source = element._source;
          var owner = element._owner;
          if (config != null) {
            if (hasValidRef(config)) {
              ref = config.ref;
              owner = ReactCurrentOwner.current;
            }
            if (hasValidKey(config)) {
              {
                checkKeyStringCoercion(config.key);
              }
              key = "" + config.key;
            }
            var defaultProps;
            if (element.type && element.type.defaultProps) {
              defaultProps = element.type.defaultProps;
            }
            for (propName in config) {
              if (hasOwnProperty.call(config, propName) && !RESERVED_PROPS.hasOwnProperty(propName)) {
                if (config[propName] === void 0 && defaultProps !== void 0) {
                  props[propName] = defaultProps[propName];
                } else {
                  props[propName] = config[propName];
                }
              }
            }
          }
          var childrenLength = arguments.length - 2;
          if (childrenLength === 1) {
            props.children = children;
          } else if (childrenLength > 1) {
            var childArray = Array(childrenLength);
            for (var i = 0; i < childrenLength; i++) {
              childArray[i] = arguments[i + 2];
            }
            props.children = childArray;
          }
          return ReactElement(element.type, key, ref, self, source, owner, props);
        }
        function isValidElement(object) {
          return typeof object === "object" && object !== null && object.$$typeof === REACT_ELEMENT_TYPE;
        }
        var SEPARATOR = ".";
        var SUBSEPARATOR = ":";
        function escape(key) {
          var escapeRegex = /[=:]/g;
          var escaperLookup = {
            "=": "=0",
            ":": "=2"
          };
          var escapedString = key.replace(escapeRegex, function(match) {
            return escaperLookup[match];
          });
          return "$" + escapedString;
        }
        var didWarnAboutMaps = false;
        var userProvidedKeyEscapeRegex = /\/+/g;
        function escapeUserProvidedKey(text) {
          return text.replace(userProvidedKeyEscapeRegex, "$&/");
        }
        function getElementKey(element, index) {
          if (typeof element === "object" && element !== null && element.key != null) {
            {
              checkKeyStringCoercion(element.key);
            }
            return escape("" + element.key);
          }
          return index.toString(36);
        }
        function mapIntoArray(children, array, escapedPrefix, nameSoFar, callback) {
          var type = typeof children;
          if (type === "undefined" || type === "boolean") {
            children = null;
          }
          var invokeCallback = false;
          if (children === null) {
            invokeCallback = true;
          } else {
            switch (type) {
              case "string":
              case "number":
                invokeCallback = true;
                break;
              case "object":
                switch (children.$$typeof) {
                  case REACT_ELEMENT_TYPE:
                  case REACT_PORTAL_TYPE:
                    invokeCallback = true;
                }
            }
          }
          if (invokeCallback) {
            var _child = children;
            var mappedChild = callback(_child);
            var childKey = nameSoFar === "" ? SEPARATOR + getElementKey(_child, 0) : nameSoFar;
            if (isArray(mappedChild)) {
              var escapedChildKey = "";
              if (childKey != null) {
                escapedChildKey = escapeUserProvidedKey(childKey) + "/";
              }
              mapIntoArray(mappedChild, array, escapedChildKey, "", function(c) {
                return c;
              });
            } else if (mappedChild != null) {
              if (isValidElement(mappedChild)) {
                {
                  if (mappedChild.key && (!_child || _child.key !== mappedChild.key)) {
                    checkKeyStringCoercion(mappedChild.key);
                  }
                }
                mappedChild = cloneAndReplaceKey(
                  mappedChild,
                  // Keep both the (mapped) and old keys if they differ, just as
                  // traverseAllChildren used to do for objects as children
                  escapedPrefix + // $FlowFixMe Flow incorrectly thinks React.Portal doesn't have a key
                  (mappedChild.key && (!_child || _child.key !== mappedChild.key) ? (
                    // $FlowFixMe Flow incorrectly thinks existing element's key can be a number
                    // eslint-disable-next-line react-internal/safe-string-coercion
                    escapeUserProvidedKey("" + mappedChild.key) + "/"
                  ) : "") + childKey
                );
              }
              array.push(mappedChild);
            }
            return 1;
          }
          var child;
          var nextName;
          var subtreeCount = 0;
          var nextNamePrefix = nameSoFar === "" ? SEPARATOR : nameSoFar + SUBSEPARATOR;
          if (isArray(children)) {
            for (var i = 0; i < children.length; i++) {
              child = children[i];
              nextName = nextNamePrefix + getElementKey(child, i);
              subtreeCount += mapIntoArray(child, array, escapedPrefix, nextName, callback);
            }
          } else {
            var iteratorFn = getIteratorFn(children);
            if (typeof iteratorFn === "function") {
              var iterableChildren = children;
              {
                if (iteratorFn === iterableChildren.entries) {
                  if (!didWarnAboutMaps) {
                    warn("Using Maps as children is not supported. Use an array of keyed ReactElements instead.");
                  }
                  didWarnAboutMaps = true;
                }
              }
              var iterator = iteratorFn.call(iterableChildren);
              var step;
              var ii = 0;
              while (!(step = iterator.next()).done) {
                child = step.value;
                nextName = nextNamePrefix + getElementKey(child, ii++);
                subtreeCount += mapIntoArray(child, array, escapedPrefix, nextName, callback);
              }
            } else if (type === "object") {
              var childrenString = String(children);
              throw new Error("Objects are not valid as a React child (found: " + (childrenString === "[object Object]" ? "object with keys {" + Object.keys(children).join(", ") + "}" : childrenString) + "). If you meant to render a collection of children, use an array instead.");
            }
          }
          return subtreeCount;
        }
        function mapChildren(children, func, context) {
          if (children == null) {
            return children;
          }
          var result = [];
          var count = 0;
          mapIntoArray(children, result, "", "", function(child) {
            return func.call(context, child, count++);
          });
          return result;
        }
        function countChildren(children) {
          var n = 0;
          mapChildren(children, function() {
            n++;
          });
          return n;
        }
        function forEachChildren(children, forEachFunc, forEachContext) {
          mapChildren(children, function() {
            forEachFunc.apply(this, arguments);
          }, forEachContext);
        }
        function toArray(children) {
          return mapChildren(children, function(child) {
            return child;
          }) || [];
        }
        function onlyChild(children) {
          if (!isValidElement(children)) {
            throw new Error("React.Children.only expected to receive a single React element child.");
          }
          return children;
        }
        function createContext(defaultValue) {
          var context = {
            $$typeof: REACT_CONTEXT_TYPE,
            // As a workaround to support multiple concurrent renderers, we categorize
            // some renderers as primary and others as secondary. We only expect
            // there to be two concurrent renderers at most: React Native (primary) and
            // Fabric (secondary); React DOM (primary) and React ART (secondary).
            // Secondary renderers store their context values on separate fields.
            _currentValue: defaultValue,
            _currentValue2: defaultValue,
            // Used to track how many concurrent renderers this context currently
            // supports within in a single renderer. Such as parallel server rendering.
            _threadCount: 0,
            // These are circular
            Provider: null,
            Consumer: null,
            // Add these to use same hidden class in VM as ServerContext
            _defaultValue: null,
            _globalName: null
          };
          context.Provider = {
            $$typeof: REACT_PROVIDER_TYPE,
            _context: context
          };
          var hasWarnedAboutUsingNestedContextConsumers = false;
          var hasWarnedAboutUsingConsumerProvider = false;
          var hasWarnedAboutDisplayNameOnConsumer = false;
          {
            var Consumer = {
              $$typeof: REACT_CONTEXT_TYPE,
              _context: context
            };
            Object.defineProperties(Consumer, {
              Provider: {
                get: function() {
                  if (!hasWarnedAboutUsingConsumerProvider) {
                    hasWarnedAboutUsingConsumerProvider = true;
                    error("Rendering <Context.Consumer.Provider> is not supported and will be removed in a future major release. Did you mean to render <Context.Provider> instead?");
                  }
                  return context.Provider;
                },
                set: function(_Provider) {
                  context.Provider = _Provider;
                }
              },
              _currentValue: {
                get: function() {
                  return context._currentValue;
                },
                set: function(_currentValue) {
                  context._currentValue = _currentValue;
                }
              },
              _currentValue2: {
                get: function() {
                  return context._currentValue2;
                },
                set: function(_currentValue2) {
                  context._currentValue2 = _currentValue2;
                }
              },
              _threadCount: {
                get: function() {
                  return context._threadCount;
                },
                set: function(_threadCount) {
                  context._threadCount = _threadCount;
                }
              },
              Consumer: {
                get: function() {
                  if (!hasWarnedAboutUsingNestedContextConsumers) {
                    hasWarnedAboutUsingNestedContextConsumers = true;
                    error("Rendering <Context.Consumer.Consumer> is not supported and will be removed in a future major release. Did you mean to render <Context.Consumer> instead?");
                  }
                  return context.Consumer;
                }
              },
              displayName: {
                get: function() {
                  return context.displayName;
                },
                set: function(displayName) {
                  if (!hasWarnedAboutDisplayNameOnConsumer) {
                    warn("Setting `displayName` on Context.Consumer has no effect. You should set it directly on the context with Context.displayName = '%s'.", displayName);
                    hasWarnedAboutDisplayNameOnConsumer = true;
                  }
                }
              }
            });
            context.Consumer = Consumer;
          }
          {
            context._currentRenderer = null;
            context._currentRenderer2 = null;
          }
          return context;
        }
        var Uninitialized = -1;
        var Pending = 0;
        var Resolved = 1;
        var Rejected = 2;
        function lazyInitializer(payload) {
          if (payload._status === Uninitialized) {
            var ctor = payload._result;
            var thenable = ctor();
            thenable.then(function(moduleObject2) {
              if (payload._status === Pending || payload._status === Uninitialized) {
                var resolved = payload;
                resolved._status = Resolved;
                resolved._result = moduleObject2;
              }
            }, function(error2) {
              if (payload._status === Pending || payload._status === Uninitialized) {
                var rejected = payload;
                rejected._status = Rejected;
                rejected._result = error2;
              }
            });
            if (payload._status === Uninitialized) {
              var pending = payload;
              pending._status = Pending;
              pending._result = thenable;
            }
          }
          if (payload._status === Resolved) {
            var moduleObject = payload._result;
            {
              if (moduleObject === void 0) {
                error("lazy: Expected the result of a dynamic import() call. Instead received: %s\n\nYour code should look like: \n  const MyComponent = lazy(() => import('./MyComponent'))\n\nDid you accidentally put curly braces around the import?", moduleObject);
              }
            }
            {
              if (!("default" in moduleObject)) {
                error("lazy: Expected the result of a dynamic import() call. Instead received: %s\n\nYour code should look like: \n  const MyComponent = lazy(() => import('./MyComponent'))", moduleObject);
              }
            }
            return moduleObject.default;
          } else {
            throw payload._result;
          }
        }
        function lazy(ctor) {
          var payload = {
            // We use these fields to store the result.
            _status: Uninitialized,
            _result: ctor
          };
          var lazyType = {
            $$typeof: REACT_LAZY_TYPE,
            _payload: payload,
            _init: lazyInitializer
          };
          {
            var defaultProps;
            var propTypes;
            Object.defineProperties(lazyType, {
              defaultProps: {
                configurable: true,
                get: function() {
                  return defaultProps;
                },
                set: function(newDefaultProps) {
                  error("React.lazy(...): It is not supported to assign `defaultProps` to a lazy component import. Either specify them where the component is defined, or create a wrapping component around it.");
                  defaultProps = newDefaultProps;
                  Object.defineProperty(lazyType, "defaultProps", {
                    enumerable: true
                  });
                }
              },
              propTypes: {
                configurable: true,
                get: function() {
                  return propTypes;
                },
                set: function(newPropTypes) {
                  error("React.lazy(...): It is not supported to assign `propTypes` to a lazy component import. Either specify them where the component is defined, or create a wrapping component around it.");
                  propTypes = newPropTypes;
                  Object.defineProperty(lazyType, "propTypes", {
                    enumerable: true
                  });
                }
              }
            });
          }
          return lazyType;
        }
        function forwardRef(render) {
          {
            if (render != null && render.$$typeof === REACT_MEMO_TYPE) {
              error("forwardRef requires a render function but received a `memo` component. Instead of forwardRef(memo(...)), use memo(forwardRef(...)).");
            } else if (typeof render !== "function") {
              error("forwardRef requires a render function but was given %s.", render === null ? "null" : typeof render);
            } else {
              if (render.length !== 0 && render.length !== 2) {
                error("forwardRef render functions accept exactly two parameters: props and ref. %s", render.length === 1 ? "Did you forget to use the ref parameter?" : "Any additional parameter will be undefined.");
              }
            }
            if (render != null) {
              if (render.defaultProps != null || render.propTypes != null) {
                error("forwardRef render functions do not support propTypes or defaultProps. Did you accidentally pass a React component?");
              }
            }
          }
          var elementType = {
            $$typeof: REACT_FORWARD_REF_TYPE,
            render
          };
          {
            var ownName;
            Object.defineProperty(elementType, "displayName", {
              enumerable: false,
              configurable: true,
              get: function() {
                return ownName;
              },
              set: function(name) {
                ownName = name;
                if (!render.name && !render.displayName) {
                  render.displayName = name;
                }
              }
            });
          }
          return elementType;
        }
        var REACT_MODULE_REFERENCE;
        {
          REACT_MODULE_REFERENCE = /* @__PURE__ */ Symbol.for("react.module.reference");
        }
        function isValidElementType(type) {
          if (typeof type === "string" || typeof type === "function") {
            return true;
          }
          if (type === REACT_FRAGMENT_TYPE || type === REACT_PROFILER_TYPE || enableDebugTracing || type === REACT_STRICT_MODE_TYPE || type === REACT_SUSPENSE_TYPE || type === REACT_SUSPENSE_LIST_TYPE || enableLegacyHidden || type === REACT_OFFSCREEN_TYPE || enableScopeAPI || enableCacheElement || enableTransitionTracing) {
            return true;
          }
          if (typeof type === "object" && type !== null) {
            if (type.$$typeof === REACT_LAZY_TYPE || type.$$typeof === REACT_MEMO_TYPE || type.$$typeof === REACT_PROVIDER_TYPE || type.$$typeof === REACT_CONTEXT_TYPE || type.$$typeof === REACT_FORWARD_REF_TYPE || // This needs to include all possible module reference object
            // types supported by any Flight configuration anywhere since
            // we don't know which Flight build this will end up being used
            // with.
            type.$$typeof === REACT_MODULE_REFERENCE || type.getModuleId !== void 0) {
              return true;
            }
          }
          return false;
        }
        function memo(type, compare) {
          {
            if (!isValidElementType(type)) {
              error("memo: The first argument must be a component. Instead received: %s", type === null ? "null" : typeof type);
            }
          }
          var elementType = {
            $$typeof: REACT_MEMO_TYPE,
            type,
            compare: compare === void 0 ? null : compare
          };
          {
            var ownName;
            Object.defineProperty(elementType, "displayName", {
              enumerable: false,
              configurable: true,
              get: function() {
                return ownName;
              },
              set: function(name) {
                ownName = name;
                if (!type.name && !type.displayName) {
                  type.displayName = name;
                }
              }
            });
          }
          return elementType;
        }
        function resolveDispatcher() {
          var dispatcher = ReactCurrentDispatcher.current;
          {
            if (dispatcher === null) {
              error("Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:\n1. You might have mismatching versions of React and the renderer (such as React DOM)\n2. You might be breaking the Rules of Hooks\n3. You might have more than one copy of React in the same app\nSee https://reactjs.org/link/invalid-hook-call for tips about how to debug and fix this problem.");
            }
          }
          return dispatcher;
        }
        function useContext(Context) {
          var dispatcher = resolveDispatcher();
          {
            if (Context._context !== void 0) {
              var realContext = Context._context;
              if (realContext.Consumer === Context) {
                error("Calling useContext(Context.Consumer) is not supported, may cause bugs, and will be removed in a future major release. Did you mean to call useContext(Context) instead?");
              } else if (realContext.Provider === Context) {
                error("Calling useContext(Context.Provider) is not supported. Did you mean to call useContext(Context) instead?");
              }
            }
          }
          return dispatcher.useContext(Context);
        }
        function useState(initialState) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useState(initialState);
        }
        function useReducer(reducer, initialArg, init) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useReducer(reducer, initialArg, init);
        }
        function useRef(initialValue) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useRef(initialValue);
        }
        function useEffect(create2, deps) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useEffect(create2, deps);
        }
        function useInsertionEffect(create2, deps) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useInsertionEffect(create2, deps);
        }
        function useLayoutEffect(create2, deps) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useLayoutEffect(create2, deps);
        }
        function useCallback(callback, deps) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useCallback(callback, deps);
        }
        function useMemo(create2, deps) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useMemo(create2, deps);
        }
        function useImperativeHandle(ref, create2, deps) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useImperativeHandle(ref, create2, deps);
        }
        function useDebugValue2(value, formatterFn) {
          {
            var dispatcher = resolveDispatcher();
            return dispatcher.useDebugValue(value, formatterFn);
          }
        }
        function useTransition() {
          var dispatcher = resolveDispatcher();
          return dispatcher.useTransition();
        }
        function useDeferredValue(value) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useDeferredValue(value);
        }
        function useId() {
          var dispatcher = resolveDispatcher();
          return dispatcher.useId();
        }
        function useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot) {
          var dispatcher = resolveDispatcher();
          return dispatcher.useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
        }
        var disabledDepth = 0;
        var prevLog;
        var prevInfo;
        var prevWarn;
        var prevError;
        var prevGroup;
        var prevGroupCollapsed;
        var prevGroupEnd;
        function disabledLog() {
        }
        disabledLog.__reactDisabledLog = true;
        function disableLogs() {
          {
            if (disabledDepth === 0) {
              prevLog = console.log;
              prevInfo = console.info;
              prevWarn = console.warn;
              prevError = console.error;
              prevGroup = console.group;
              prevGroupCollapsed = console.groupCollapsed;
              prevGroupEnd = console.groupEnd;
              var props = {
                configurable: true,
                enumerable: true,
                value: disabledLog,
                writable: true
              };
              Object.defineProperties(console, {
                info: props,
                log: props,
                warn: props,
                error: props,
                group: props,
                groupCollapsed: props,
                groupEnd: props
              });
            }
            disabledDepth++;
          }
        }
        function reenableLogs() {
          {
            disabledDepth--;
            if (disabledDepth === 0) {
              var props = {
                configurable: true,
                enumerable: true,
                writable: true
              };
              Object.defineProperties(console, {
                log: assign({}, props, {
                  value: prevLog
                }),
                info: assign({}, props, {
                  value: prevInfo
                }),
                warn: assign({}, props, {
                  value: prevWarn
                }),
                error: assign({}, props, {
                  value: prevError
                }),
                group: assign({}, props, {
                  value: prevGroup
                }),
                groupCollapsed: assign({}, props, {
                  value: prevGroupCollapsed
                }),
                groupEnd: assign({}, props, {
                  value: prevGroupEnd
                })
              });
            }
            if (disabledDepth < 0) {
              error("disabledDepth fell below zero. This is a bug in React. Please file an issue.");
            }
          }
        }
        var ReactCurrentDispatcher$1 = ReactSharedInternals.ReactCurrentDispatcher;
        var prefix;
        function describeBuiltInComponentFrame(name, source, ownerFn) {
          {
            if (prefix === void 0) {
              try {
                throw Error();
              } catch (x) {
                var match = x.stack.trim().match(/\n( *(at )?)/);
                prefix = match && match[1] || "";
              }
            }
            return "\n" + prefix + name;
          }
        }
        var reentry = false;
        var componentFrameCache;
        {
          var PossiblyWeakMap = typeof WeakMap === "function" ? WeakMap : Map;
          componentFrameCache = new PossiblyWeakMap();
        }
        function describeNativeComponentFrame(fn, construct) {
          if (!fn || reentry) {
            return "";
          }
          {
            var frame = componentFrameCache.get(fn);
            if (frame !== void 0) {
              return frame;
            }
          }
          var control;
          reentry = true;
          var previousPrepareStackTrace = Error.prepareStackTrace;
          Error.prepareStackTrace = void 0;
          var previousDispatcher;
          {
            previousDispatcher = ReactCurrentDispatcher$1.current;
            ReactCurrentDispatcher$1.current = null;
            disableLogs();
          }
          try {
            if (construct) {
              var Fake = function() {
                throw Error();
              };
              Object.defineProperty(Fake.prototype, "props", {
                set: function() {
                  throw Error();
                }
              });
              if (typeof Reflect === "object" && Reflect.construct) {
                try {
                  Reflect.construct(Fake, []);
                } catch (x) {
                  control = x;
                }
                Reflect.construct(fn, [], Fake);
              } else {
                try {
                  Fake.call();
                } catch (x) {
                  control = x;
                }
                fn.call(Fake.prototype);
              }
            } else {
              try {
                throw Error();
              } catch (x) {
                control = x;
              }
              fn();
            }
          } catch (sample) {
            if (sample && control && typeof sample.stack === "string") {
              var sampleLines = sample.stack.split("\n");
              var controlLines = control.stack.split("\n");
              var s = sampleLines.length - 1;
              var c = controlLines.length - 1;
              while (s >= 1 && c >= 0 && sampleLines[s] !== controlLines[c]) {
                c--;
              }
              for (; s >= 1 && c >= 0; s--, c--) {
                if (sampleLines[s] !== controlLines[c]) {
                  if (s !== 1 || c !== 1) {
                    do {
                      s--;
                      c--;
                      if (c < 0 || sampleLines[s] !== controlLines[c]) {
                        var _frame = "\n" + sampleLines[s].replace(" at new ", " at ");
                        if (fn.displayName && _frame.includes("<anonymous>")) {
                          _frame = _frame.replace("<anonymous>", fn.displayName);
                        }
                        {
                          if (typeof fn === "function") {
                            componentFrameCache.set(fn, _frame);
                          }
                        }
                        return _frame;
                      }
                    } while (s >= 1 && c >= 0);
                  }
                  break;
                }
              }
            }
          } finally {
            reentry = false;
            {
              ReactCurrentDispatcher$1.current = previousDispatcher;
              reenableLogs();
            }
            Error.prepareStackTrace = previousPrepareStackTrace;
          }
          var name = fn ? fn.displayName || fn.name : "";
          var syntheticFrame = name ? describeBuiltInComponentFrame(name) : "";
          {
            if (typeof fn === "function") {
              componentFrameCache.set(fn, syntheticFrame);
            }
          }
          return syntheticFrame;
        }
        function describeFunctionComponentFrame(fn, source, ownerFn) {
          {
            return describeNativeComponentFrame(fn, false);
          }
        }
        function shouldConstruct(Component2) {
          var prototype = Component2.prototype;
          return !!(prototype && prototype.isReactComponent);
        }
        function describeUnknownElementTypeFrameInDEV(type, source, ownerFn) {
          if (type == null) {
            return "";
          }
          if (typeof type === "function") {
            {
              return describeNativeComponentFrame(type, shouldConstruct(type));
            }
          }
          if (typeof type === "string") {
            return describeBuiltInComponentFrame(type);
          }
          switch (type) {
            case REACT_SUSPENSE_TYPE:
              return describeBuiltInComponentFrame("Suspense");
            case REACT_SUSPENSE_LIST_TYPE:
              return describeBuiltInComponentFrame("SuspenseList");
          }
          if (typeof type === "object") {
            switch (type.$$typeof) {
              case REACT_FORWARD_REF_TYPE:
                return describeFunctionComponentFrame(type.render);
              case REACT_MEMO_TYPE:
                return describeUnknownElementTypeFrameInDEV(type.type, source, ownerFn);
              case REACT_LAZY_TYPE: {
                var lazyComponent = type;
                var payload = lazyComponent._payload;
                var init = lazyComponent._init;
                try {
                  return describeUnknownElementTypeFrameInDEV(init(payload), source, ownerFn);
                } catch (x) {
                }
              }
            }
          }
          return "";
        }
        var loggedTypeFailures = {};
        var ReactDebugCurrentFrame$1 = ReactSharedInternals.ReactDebugCurrentFrame;
        function setCurrentlyValidatingElement(element) {
          {
            if (element) {
              var owner = element._owner;
              var stack = describeUnknownElementTypeFrameInDEV(element.type, element._source, owner ? owner.type : null);
              ReactDebugCurrentFrame$1.setExtraStackFrame(stack);
            } else {
              ReactDebugCurrentFrame$1.setExtraStackFrame(null);
            }
          }
        }
        function checkPropTypes(typeSpecs, values, location, componentName, element) {
          {
            var has = Function.call.bind(hasOwnProperty);
            for (var typeSpecName in typeSpecs) {
              if (has(typeSpecs, typeSpecName)) {
                var error$1 = void 0;
                try {
                  if (typeof typeSpecs[typeSpecName] !== "function") {
                    var err = Error((componentName || "React class") + ": " + location + " type `" + typeSpecName + "` is invalid; it must be a function, usually from the `prop-types` package, but received `" + typeof typeSpecs[typeSpecName] + "`.This often happens because of typos such as `PropTypes.function` instead of `PropTypes.func`.");
                    err.name = "Invariant Violation";
                    throw err;
                  }
                  error$1 = typeSpecs[typeSpecName](values, typeSpecName, componentName, location, null, "SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED");
                } catch (ex) {
                  error$1 = ex;
                }
                if (error$1 && !(error$1 instanceof Error)) {
                  setCurrentlyValidatingElement(element);
                  error("%s: type specification of %s `%s` is invalid; the type checker function must return `null` or an `Error` but returned a %s. You may have forgotten to pass an argument to the type checker creator (arrayOf, instanceOf, objectOf, oneOf, oneOfType, and shape all require an argument).", componentName || "React class", location, typeSpecName, typeof error$1);
                  setCurrentlyValidatingElement(null);
                }
                if (error$1 instanceof Error && !(error$1.message in loggedTypeFailures)) {
                  loggedTypeFailures[error$1.message] = true;
                  setCurrentlyValidatingElement(element);
                  error("Failed %s type: %s", location, error$1.message);
                  setCurrentlyValidatingElement(null);
                }
              }
            }
          }
        }
        function setCurrentlyValidatingElement$1(element) {
          {
            if (element) {
              var owner = element._owner;
              var stack = describeUnknownElementTypeFrameInDEV(element.type, element._source, owner ? owner.type : null);
              setExtraStackFrame(stack);
            } else {
              setExtraStackFrame(null);
            }
          }
        }
        var propTypesMisspellWarningShown;
        {
          propTypesMisspellWarningShown = false;
        }
        function getDeclarationErrorAddendum() {
          if (ReactCurrentOwner.current) {
            var name = getComponentNameFromType(ReactCurrentOwner.current.type);
            if (name) {
              return "\n\nCheck the render method of `" + name + "`.";
            }
          }
          return "";
        }
        function getSourceInfoErrorAddendum(source) {
          if (source !== void 0) {
            var fileName = source.fileName.replace(/^.*[\\\/]/, "");
            var lineNumber = source.lineNumber;
            return "\n\nCheck your code at " + fileName + ":" + lineNumber + ".";
          }
          return "";
        }
        function getSourceInfoErrorAddendumForProps(elementProps) {
          if (elementProps !== null && elementProps !== void 0) {
            return getSourceInfoErrorAddendum(elementProps.__source);
          }
          return "";
        }
        var ownerHasKeyUseWarning = {};
        function getCurrentComponentErrorInfo(parentType) {
          var info = getDeclarationErrorAddendum();
          if (!info) {
            var parentName = typeof parentType === "string" ? parentType : parentType.displayName || parentType.name;
            if (parentName) {
              info = "\n\nCheck the top-level render call using <" + parentName + ">.";
            }
          }
          return info;
        }
        function validateExplicitKey(element, parentType) {
          if (!element._store || element._store.validated || element.key != null) {
            return;
          }
          element._store.validated = true;
          var currentComponentErrorInfo = getCurrentComponentErrorInfo(parentType);
          if (ownerHasKeyUseWarning[currentComponentErrorInfo]) {
            return;
          }
          ownerHasKeyUseWarning[currentComponentErrorInfo] = true;
          var childOwner = "";
          if (element && element._owner && element._owner !== ReactCurrentOwner.current) {
            childOwner = " It was passed a child from " + getComponentNameFromType(element._owner.type) + ".";
          }
          {
            setCurrentlyValidatingElement$1(element);
            error('Each child in a list should have a unique "key" prop.%s%s See https://reactjs.org/link/warning-keys for more information.', currentComponentErrorInfo, childOwner);
            setCurrentlyValidatingElement$1(null);
          }
        }
        function validateChildKeys(node, parentType) {
          if (typeof node !== "object") {
            return;
          }
          if (isArray(node)) {
            for (var i = 0; i < node.length; i++) {
              var child = node[i];
              if (isValidElement(child)) {
                validateExplicitKey(child, parentType);
              }
            }
          } else if (isValidElement(node)) {
            if (node._store) {
              node._store.validated = true;
            }
          } else if (node) {
            var iteratorFn = getIteratorFn(node);
            if (typeof iteratorFn === "function") {
              if (iteratorFn !== node.entries) {
                var iterator = iteratorFn.call(node);
                var step;
                while (!(step = iterator.next()).done) {
                  if (isValidElement(step.value)) {
                    validateExplicitKey(step.value, parentType);
                  }
                }
              }
            }
          }
        }
        function validatePropTypes(element) {
          {
            var type = element.type;
            if (type === null || type === void 0 || typeof type === "string") {
              return;
            }
            var propTypes;
            if (typeof type === "function") {
              propTypes = type.propTypes;
            } else if (typeof type === "object" && (type.$$typeof === REACT_FORWARD_REF_TYPE || // Note: Memo only checks outer props here.
            // Inner props are checked in the reconciler.
            type.$$typeof === REACT_MEMO_TYPE)) {
              propTypes = type.propTypes;
            } else {
              return;
            }
            if (propTypes) {
              var name = getComponentNameFromType(type);
              checkPropTypes(propTypes, element.props, "prop", name, element);
            } else if (type.PropTypes !== void 0 && !propTypesMisspellWarningShown) {
              propTypesMisspellWarningShown = true;
              var _name = getComponentNameFromType(type);
              error("Component %s declared `PropTypes` instead of `propTypes`. Did you misspell the property assignment?", _name || "Unknown");
            }
            if (typeof type.getDefaultProps === "function" && !type.getDefaultProps.isReactClassApproved) {
              error("getDefaultProps is only used on classic React.createClass definitions. Use a static property named `defaultProps` instead.");
            }
          }
        }
        function validateFragmentProps(fragment) {
          {
            var keys = Object.keys(fragment.props);
            for (var i = 0; i < keys.length; i++) {
              var key = keys[i];
              if (key !== "children" && key !== "key") {
                setCurrentlyValidatingElement$1(fragment);
                error("Invalid prop `%s` supplied to `React.Fragment`. React.Fragment can only have `key` and `children` props.", key);
                setCurrentlyValidatingElement$1(null);
                break;
              }
            }
            if (fragment.ref !== null) {
              setCurrentlyValidatingElement$1(fragment);
              error("Invalid attribute `ref` supplied to `React.Fragment`.");
              setCurrentlyValidatingElement$1(null);
            }
          }
        }
        function createElementWithValidation(type, props, children) {
          var validType = isValidElementType(type);
          if (!validType) {
            var info = "";
            if (type === void 0 || typeof type === "object" && type !== null && Object.keys(type).length === 0) {
              info += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports.";
            }
            var sourceInfo = getSourceInfoErrorAddendumForProps(props);
            if (sourceInfo) {
              info += sourceInfo;
            } else {
              info += getDeclarationErrorAddendum();
            }
            var typeString;
            if (type === null) {
              typeString = "null";
            } else if (isArray(type)) {
              typeString = "array";
            } else if (type !== void 0 && type.$$typeof === REACT_ELEMENT_TYPE) {
              typeString = "<" + (getComponentNameFromType(type.type) || "Unknown") + " />";
              info = " Did you accidentally export a JSX literal instead of a component?";
            } else {
              typeString = typeof type;
            }
            {
              error("React.createElement: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s", typeString, info);
            }
          }
          var element = createElement.apply(this, arguments);
          if (element == null) {
            return element;
          }
          if (validType) {
            for (var i = 2; i < arguments.length; i++) {
              validateChildKeys(arguments[i], type);
            }
          }
          if (type === REACT_FRAGMENT_TYPE) {
            validateFragmentProps(element);
          } else {
            validatePropTypes(element);
          }
          return element;
        }
        var didWarnAboutDeprecatedCreateFactory = false;
        function createFactoryWithValidation(type) {
          var validatedFactory = createElementWithValidation.bind(null, type);
          validatedFactory.type = type;
          {
            if (!didWarnAboutDeprecatedCreateFactory) {
              didWarnAboutDeprecatedCreateFactory = true;
              warn("React.createFactory() is deprecated and will be removed in a future major release. Consider using JSX or use React.createElement() directly instead.");
            }
            Object.defineProperty(validatedFactory, "type", {
              enumerable: false,
              get: function() {
                warn("Factory.type is deprecated. Access the class directly before passing it to createFactory.");
                Object.defineProperty(this, "type", {
                  value: type
                });
                return type;
              }
            });
          }
          return validatedFactory;
        }
        function cloneElementWithValidation(element, props, children) {
          var newElement = cloneElement.apply(this, arguments);
          for (var i = 2; i < arguments.length; i++) {
            validateChildKeys(arguments[i], newElement.type);
          }
          validatePropTypes(newElement);
          return newElement;
        }
        function startTransition(scope, options) {
          var prevTransition = ReactCurrentBatchConfig.transition;
          ReactCurrentBatchConfig.transition = {};
          var currentTransition = ReactCurrentBatchConfig.transition;
          {
            ReactCurrentBatchConfig.transition._updatedFibers = /* @__PURE__ */ new Set();
          }
          try {
            scope();
          } finally {
            ReactCurrentBatchConfig.transition = prevTransition;
            {
              if (prevTransition === null && currentTransition._updatedFibers) {
                var updatedFibersCount = currentTransition._updatedFibers.size;
                if (updatedFibersCount > 10) {
                  warn("Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table.");
                }
                currentTransition._updatedFibers.clear();
              }
            }
          }
        }
        var didWarnAboutMessageChannel = false;
        var enqueueTaskImpl = null;
        function enqueueTask(task) {
          if (enqueueTaskImpl === null) {
            try {
              var requireString = ("require" + Math.random()).slice(0, 7);
              var nodeRequire = module && module[requireString];
              enqueueTaskImpl = nodeRequire.call(module, "timers").setImmediate;
            } catch (_err) {
              enqueueTaskImpl = function(callback) {
                {
                  if (didWarnAboutMessageChannel === false) {
                    didWarnAboutMessageChannel = true;
                    if (typeof MessageChannel === "undefined") {
                      error("This browser does not have a MessageChannel implementation, so enqueuing tasks via await act(async () => ...) will fail. Please file an issue at https://github.com/facebook/react/issues if you encounter this warning.");
                    }
                  }
                }
                var channel = new MessageChannel();
                channel.port1.onmessage = callback;
                channel.port2.postMessage(void 0);
              };
            }
          }
          return enqueueTaskImpl(task);
        }
        var actScopeDepth = 0;
        var didWarnNoAwaitAct = false;
        function act(callback) {
          {
            var prevActScopeDepth = actScopeDepth;
            actScopeDepth++;
            if (ReactCurrentActQueue.current === null) {
              ReactCurrentActQueue.current = [];
            }
            var prevIsBatchingLegacy = ReactCurrentActQueue.isBatchingLegacy;
            var result;
            try {
              ReactCurrentActQueue.isBatchingLegacy = true;
              result = callback();
              if (!prevIsBatchingLegacy && ReactCurrentActQueue.didScheduleLegacyUpdate) {
                var queue = ReactCurrentActQueue.current;
                if (queue !== null) {
                  ReactCurrentActQueue.didScheduleLegacyUpdate = false;
                  flushActQueue(queue);
                }
              }
            } catch (error2) {
              popActScope(prevActScopeDepth);
              throw error2;
            } finally {
              ReactCurrentActQueue.isBatchingLegacy = prevIsBatchingLegacy;
            }
            if (result !== null && typeof result === "object" && typeof result.then === "function") {
              var thenableResult = result;
              var wasAwaited = false;
              var thenable = {
                then: function(resolve, reject) {
                  wasAwaited = true;
                  thenableResult.then(function(returnValue2) {
                    popActScope(prevActScopeDepth);
                    if (actScopeDepth === 0) {
                      recursivelyFlushAsyncActWork(returnValue2, resolve, reject);
                    } else {
                      resolve(returnValue2);
                    }
                  }, function(error2) {
                    popActScope(prevActScopeDepth);
                    reject(error2);
                  });
                }
              };
              {
                if (!didWarnNoAwaitAct && typeof Promise !== "undefined") {
                  Promise.resolve().then(function() {
                  }).then(function() {
                    if (!wasAwaited) {
                      didWarnNoAwaitAct = true;
                      error("You called act(async () => ...) without await. This could lead to unexpected testing behaviour, interleaving multiple act calls and mixing their scopes. You should - await act(async () => ...);");
                    }
                  });
                }
              }
              return thenable;
            } else {
              var returnValue = result;
              popActScope(prevActScopeDepth);
              if (actScopeDepth === 0) {
                var _queue = ReactCurrentActQueue.current;
                if (_queue !== null) {
                  flushActQueue(_queue);
                  ReactCurrentActQueue.current = null;
                }
                var _thenable = {
                  then: function(resolve, reject) {
                    if (ReactCurrentActQueue.current === null) {
                      ReactCurrentActQueue.current = [];
                      recursivelyFlushAsyncActWork(returnValue, resolve, reject);
                    } else {
                      resolve(returnValue);
                    }
                  }
                };
                return _thenable;
              } else {
                var _thenable2 = {
                  then: function(resolve, reject) {
                    resolve(returnValue);
                  }
                };
                return _thenable2;
              }
            }
          }
        }
        function popActScope(prevActScopeDepth) {
          {
            if (prevActScopeDepth !== actScopeDepth - 1) {
              error("You seem to have overlapping act() calls, this is not supported. Be sure to await previous act() calls before making a new one. ");
            }
            actScopeDepth = prevActScopeDepth;
          }
        }
        function recursivelyFlushAsyncActWork(returnValue, resolve, reject) {
          {
            var queue = ReactCurrentActQueue.current;
            if (queue !== null) {
              try {
                flushActQueue(queue);
                enqueueTask(function() {
                  if (queue.length === 0) {
                    ReactCurrentActQueue.current = null;
                    resolve(returnValue);
                  } else {
                    recursivelyFlushAsyncActWork(returnValue, resolve, reject);
                  }
                });
              } catch (error2) {
                reject(error2);
              }
            } else {
              resolve(returnValue);
            }
          }
        }
        var isFlushing = false;
        function flushActQueue(queue) {
          {
            if (!isFlushing) {
              isFlushing = true;
              var i = 0;
              try {
                for (; i < queue.length; i++) {
                  var callback = queue[i];
                  do {
                    callback = callback(true);
                  } while (callback !== null);
                }
                queue.length = 0;
              } catch (error2) {
                queue = queue.slice(i + 1);
                throw error2;
              } finally {
                isFlushing = false;
              }
            }
          }
        }
        var createElement$1 = createElementWithValidation;
        var cloneElement$1 = cloneElementWithValidation;
        var createFactory = createFactoryWithValidation;
        var Children = {
          map: mapChildren,
          forEach: forEachChildren,
          count: countChildren,
          toArray,
          only: onlyChild
        };
        exports.Children = Children;
        exports.Component = Component;
        exports.Fragment = REACT_FRAGMENT_TYPE;
        exports.Profiler = REACT_PROFILER_TYPE;
        exports.PureComponent = PureComponent;
        exports.StrictMode = REACT_STRICT_MODE_TYPE;
        exports.Suspense = REACT_SUSPENSE_TYPE;
        exports.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = ReactSharedInternals;
        exports.act = act;
        exports.cloneElement = cloneElement$1;
        exports.createContext = createContext;
        exports.createElement = createElement$1;
        exports.createFactory = createFactory;
        exports.createRef = createRef;
        exports.forwardRef = forwardRef;
        exports.isValidElement = isValidElement;
        exports.lazy = lazy;
        exports.memo = memo;
        exports.startTransition = startTransition;
        exports.unstable_act = act;
        exports.useCallback = useCallback;
        exports.useContext = useContext;
        exports.useDebugValue = useDebugValue2;
        exports.useDeferredValue = useDeferredValue;
        exports.useEffect = useEffect;
        exports.useId = useId;
        exports.useImperativeHandle = useImperativeHandle;
        exports.useInsertionEffect = useInsertionEffect;
        exports.useLayoutEffect = useLayoutEffect;
        exports.useMemo = useMemo;
        exports.useReducer = useReducer;
        exports.useRef = useRef;
        exports.useState = useState;
        exports.useSyncExternalStore = useSyncExternalStore;
        exports.useTransition = useTransition;
        exports.version = ReactVersion;
        if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ !== "undefined" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop === "function") {
          __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(new Error());
        }
      })();
    }
  }
});

// ../Neurarch/node_modules/react/index.js
var require_react = __commonJS({
  "../Neurarch/node_modules/react/index.js"(exports, module) {
    "use strict";
    if (process.env.NODE_ENV === "production") {
      module.exports = require_react_production_min();
    } else {
      module.exports = require_react_development();
    }
  }
});

// ../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim.production.js
var require_use_sync_external_store_shim_production = __commonJS({
  "../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim.production.js"(exports) {
    "use strict";
    var React = require_react();
    function is(x, y) {
      return x === y && (0 !== x || 1 / x === 1 / y) || x !== x && y !== y;
    }
    var objectIs = "function" === typeof Object.is ? Object.is : is;
    var useState = React.useState;
    var useEffect = React.useEffect;
    var useLayoutEffect = React.useLayoutEffect;
    var useDebugValue2 = React.useDebugValue;
    function useSyncExternalStore$2(subscribe, getSnapshot) {
      var value = getSnapshot(), _useState = useState({ inst: { value, getSnapshot } }), inst = _useState[0].inst, forceUpdate = _useState[1];
      useLayoutEffect(
        function() {
          inst.value = value;
          inst.getSnapshot = getSnapshot;
          checkIfSnapshotChanged(inst) && forceUpdate({ inst });
        },
        [subscribe, value, getSnapshot]
      );
      useEffect(
        function() {
          checkIfSnapshotChanged(inst) && forceUpdate({ inst });
          return subscribe(function() {
            checkIfSnapshotChanged(inst) && forceUpdate({ inst });
          });
        },
        [subscribe]
      );
      useDebugValue2(value);
      return value;
    }
    function checkIfSnapshotChanged(inst) {
      var latestGetSnapshot = inst.getSnapshot;
      inst = inst.value;
      try {
        var nextValue = latestGetSnapshot();
        return !objectIs(inst, nextValue);
      } catch (error) {
        return true;
      }
    }
    function useSyncExternalStore$1(subscribe, getSnapshot) {
      return getSnapshot();
    }
    var shim = "undefined" === typeof window || "undefined" === typeof window.document || "undefined" === typeof window.document.createElement ? useSyncExternalStore$1 : useSyncExternalStore$2;
    exports.useSyncExternalStore = void 0 !== React.useSyncExternalStore ? React.useSyncExternalStore : shim;
  }
});

// ../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim.development.js
var require_use_sync_external_store_shim_development = __commonJS({
  "../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim.development.js"(exports) {
    "use strict";
    "production" !== process.env.NODE_ENV && (function() {
      function is(x, y) {
        return x === y && (0 !== x || 1 / x === 1 / y) || x !== x && y !== y;
      }
      function useSyncExternalStore$2(subscribe, getSnapshot) {
        didWarnOld18Alpha || void 0 === React.startTransition || (didWarnOld18Alpha = true, console.error(
          "You are using an outdated, pre-release alpha of React 18 that does not support useSyncExternalStore. The use-sync-external-store shim will not work correctly. Upgrade to a newer pre-release."
        ));
        var value = getSnapshot();
        if (!didWarnUncachedGetSnapshot) {
          var cachedValue = getSnapshot();
          objectIs(value, cachedValue) || (console.error(
            "The result of getSnapshot should be cached to avoid an infinite loop"
          ), didWarnUncachedGetSnapshot = true);
        }
        cachedValue = useState({
          inst: { value, getSnapshot }
        });
        var inst = cachedValue[0].inst, forceUpdate = cachedValue[1];
        useLayoutEffect(
          function() {
            inst.value = value;
            inst.getSnapshot = getSnapshot;
            checkIfSnapshotChanged(inst) && forceUpdate({ inst });
          },
          [subscribe, value, getSnapshot]
        );
        useEffect(
          function() {
            checkIfSnapshotChanged(inst) && forceUpdate({ inst });
            return subscribe(function() {
              checkIfSnapshotChanged(inst) && forceUpdate({ inst });
            });
          },
          [subscribe]
        );
        useDebugValue2(value);
        return value;
      }
      function checkIfSnapshotChanged(inst) {
        var latestGetSnapshot = inst.getSnapshot;
        inst = inst.value;
        try {
          var nextValue = latestGetSnapshot();
          return !objectIs(inst, nextValue);
        } catch (error) {
          return true;
        }
      }
      function useSyncExternalStore$1(subscribe, getSnapshot) {
        return getSnapshot();
      }
      "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ && "function" === typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
      var React = require_react(), objectIs = "function" === typeof Object.is ? Object.is : is, useState = React.useState, useEffect = React.useEffect, useLayoutEffect = React.useLayoutEffect, useDebugValue2 = React.useDebugValue, didWarnOld18Alpha = false, didWarnUncachedGetSnapshot = false, shim = "undefined" === typeof window || "undefined" === typeof window.document || "undefined" === typeof window.document.createElement ? useSyncExternalStore$1 : useSyncExternalStore$2;
      exports.useSyncExternalStore = void 0 !== React.useSyncExternalStore ? React.useSyncExternalStore : shim;
      "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ && "function" === typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    })();
  }
});

// ../Neurarch/node_modules/use-sync-external-store/shim/index.js
var require_shim = __commonJS({
  "../Neurarch/node_modules/use-sync-external-store/shim/index.js"(exports, module) {
    "use strict";
    if (process.env.NODE_ENV === "production") {
      module.exports = require_use_sync_external_store_shim_production();
    } else {
      module.exports = require_use_sync_external_store_shim_development();
    }
  }
});

// ../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim/with-selector.production.js
var require_with_selector_production = __commonJS({
  "../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim/with-selector.production.js"(exports) {
    "use strict";
    var React = require_react();
    var shim = require_shim();
    function is(x, y) {
      return x === y && (0 !== x || 1 / x === 1 / y) || x !== x && y !== y;
    }
    var objectIs = "function" === typeof Object.is ? Object.is : is;
    var useSyncExternalStore = shim.useSyncExternalStore;
    var useRef = React.useRef;
    var useEffect = React.useEffect;
    var useMemo = React.useMemo;
    var useDebugValue2 = React.useDebugValue;
    exports.useSyncExternalStoreWithSelector = function(subscribe, getSnapshot, getServerSnapshot, selector, isEqual) {
      var instRef = useRef(null);
      if (null === instRef.current) {
        var inst = { hasValue: false, value: null };
        instRef.current = inst;
      } else inst = instRef.current;
      instRef = useMemo(
        function() {
          function memoizedSelector(nextSnapshot) {
            if (!hasMemo) {
              hasMemo = true;
              memoizedSnapshot = nextSnapshot;
              nextSnapshot = selector(nextSnapshot);
              if (void 0 !== isEqual && inst.hasValue) {
                var currentSelection = inst.value;
                if (isEqual(currentSelection, nextSnapshot))
                  return memoizedSelection = currentSelection;
              }
              return memoizedSelection = nextSnapshot;
            }
            currentSelection = memoizedSelection;
            if (objectIs(memoizedSnapshot, nextSnapshot)) return currentSelection;
            var nextSelection = selector(nextSnapshot);
            if (void 0 !== isEqual && isEqual(currentSelection, nextSelection))
              return memoizedSnapshot = nextSnapshot, currentSelection;
            memoizedSnapshot = nextSnapshot;
            return memoizedSelection = nextSelection;
          }
          var hasMemo = false, memoizedSnapshot, memoizedSelection, maybeGetServerSnapshot = void 0 === getServerSnapshot ? null : getServerSnapshot;
          return [
            function() {
              return memoizedSelector(getSnapshot());
            },
            null === maybeGetServerSnapshot ? void 0 : function() {
              return memoizedSelector(maybeGetServerSnapshot());
            }
          ];
        },
        [getSnapshot, getServerSnapshot, selector, isEqual]
      );
      var value = useSyncExternalStore(subscribe, instRef[0], instRef[1]);
      useEffect(
        function() {
          inst.hasValue = true;
          inst.value = value;
        },
        [value]
      );
      useDebugValue2(value);
      return value;
    };
  }
});

// ../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim/with-selector.development.js
var require_with_selector_development = __commonJS({
  "../Neurarch/node_modules/use-sync-external-store/cjs/use-sync-external-store-shim/with-selector.development.js"(exports) {
    "use strict";
    "production" !== process.env.NODE_ENV && (function() {
      function is(x, y) {
        return x === y && (0 !== x || 1 / x === 1 / y) || x !== x && y !== y;
      }
      "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ && "function" === typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
      var React = require_react(), shim = require_shim(), objectIs = "function" === typeof Object.is ? Object.is : is, useSyncExternalStore = shim.useSyncExternalStore, useRef = React.useRef, useEffect = React.useEffect, useMemo = React.useMemo, useDebugValue2 = React.useDebugValue;
      exports.useSyncExternalStoreWithSelector = function(subscribe, getSnapshot, getServerSnapshot, selector, isEqual) {
        var instRef = useRef(null);
        if (null === instRef.current) {
          var inst = { hasValue: false, value: null };
          instRef.current = inst;
        } else inst = instRef.current;
        instRef = useMemo(
          function() {
            function memoizedSelector(nextSnapshot) {
              if (!hasMemo) {
                hasMemo = true;
                memoizedSnapshot = nextSnapshot;
                nextSnapshot = selector(nextSnapshot);
                if (void 0 !== isEqual && inst.hasValue) {
                  var currentSelection = inst.value;
                  if (isEqual(currentSelection, nextSnapshot))
                    return memoizedSelection = currentSelection;
                }
                return memoizedSelection = nextSnapshot;
              }
              currentSelection = memoizedSelection;
              if (objectIs(memoizedSnapshot, nextSnapshot))
                return currentSelection;
              var nextSelection = selector(nextSnapshot);
              if (void 0 !== isEqual && isEqual(currentSelection, nextSelection))
                return memoizedSnapshot = nextSnapshot, currentSelection;
              memoizedSnapshot = nextSnapshot;
              return memoizedSelection = nextSelection;
            }
            var hasMemo = false, memoizedSnapshot, memoizedSelection, maybeGetServerSnapshot = void 0 === getServerSnapshot ? null : getServerSnapshot;
            return [
              function() {
                return memoizedSelector(getSnapshot());
              },
              null === maybeGetServerSnapshot ? void 0 : function() {
                return memoizedSelector(maybeGetServerSnapshot());
              }
            ];
          },
          [getSnapshot, getServerSnapshot, selector, isEqual]
        );
        var value = useSyncExternalStore(subscribe, instRef[0], instRef[1]);
        useEffect(
          function() {
            inst.hasValue = true;
            inst.value = value;
          },
          [value]
        );
        useDebugValue2(value);
        return value;
      };
      "undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ && "function" === typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    })();
  }
});

// ../Neurarch/node_modules/use-sync-external-store/shim/with-selector.js
var require_with_selector = __commonJS({
  "../Neurarch/node_modules/use-sync-external-store/shim/with-selector.js"(exports, module) {
    "use strict";
    if (process.env.NODE_ENV === "production") {
      module.exports = require_with_selector_production();
    } else {
      module.exports = require_with_selector_development();
    }
  }
});

// src/utils/graphIO.ts
function rebuildNodeIO(components, connections) {
  const incoming = /* @__PURE__ */ new Map();
  const outgoing = /* @__PURE__ */ new Map();
  for (const conn of connections) {
    let outs = outgoing.get(conn.from);
    if (!outs) outgoing.set(conn.from, outs = []);
    outs.push(conn.to);
    let ins = incoming.get(conn.to);
    if (!ins) incoming.set(conn.to, ins = []);
    ins.push(conn.from);
  }
  for (const comp of components) {
    comp.inputs = incoming.get(comp.id) ?? [];
    comp.outputs = outgoing.get(comp.id) ?? [];
  }
  return components;
}

// src/components/MLComponents/componentRegistry.ts
var convDim = (v, i, fallback) => {
  const raw = Array.isArray(v) ? v[i] ?? v[0] : v;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
};
var convOutLen = (len, params, i) => {
  const k = convDim(params.kernelSize, i, 3);
  const s = convDim(params.stride, i, 1);
  const p = convDim(params.padding, i, 0);
  const d = convDim(params.dilation, i, 1);
  return Math.floor((len + 2 * p - d * (k - 1) - 1) / s + 1);
};
var conv2dOutputShape = (inputShape, params) => {
  if (inputShape.length >= 3) {
    const [, h, w] = inputShape;
    const { outChannels } = params;
    return [outChannels, convOutLen(h, params, 0), convOutLen(w, params, 1)];
  }
  return inputShape;
};
var conv1dOutputShape = (inputShape, params) => {
  if (inputShape.length >= 2) {
    const l = inputShape[inputShape.length - 1];
    const { outChannels } = params;
    return [...inputShape.slice(0, -2), outChannels, convOutLen(l, params, 0)];
  }
  return inputShape;
};
var pool2dOutputShape = (inputShape, params) => {
  if (inputShape.length >= 3) {
    const [c, h, w] = inputShape;
    const p = { ...params, stride: params.stride ?? params.kernelSize };
    return [c, convOutLen(h, p, 0), convOutLen(w, p, 1)];
  }
  return inputShape;
};
var recurrentOutputShape = (inputShape, params, multiplier) => {
  const hidden = (params.hiddenSize || 128) * multiplier;
  if (params.returnSequences === false) return [hidden];
  if (inputShape.length >= 3) return [inputShape[0], inputShape[1], hidden];
  return [inputShape[0] || 1, hidden];
};
var componentRegistry = {
  // ========== Basic ==========
  input: {
    type: "input",
    name: "Input",
    icon: "\u{1F4E5}",
    category: "basic",
    defaultParams: { shape: [1, 28, 28] },
    computeOutputShape: (_, params) => params.shape || [1]
  },
  output: {
    type: "output",
    name: "Output",
    icon: "\u{1F4E4}",
    category: "basic",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  linear: {
    type: "linear",
    name: "Linear",
    icon: "\u26A1",
    category: "basic",
    defaultParams: { outFeatures: 128 },
    // nn.Linear maps the LAST dim only; leading dims (seq, tokens, ...) pass
    // through. Collapsing to [outFeatures] used to fire false blocking
    // merge-shape mismatches on residual + projection patterns.
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [...inputShape.slice(0, -1), params.outFeatures];
      }
      return [params.outFeatures];
    }
  },
  flatten: {
    type: "flatten",
    name: "Flatten",
    icon: "\u{1F4C4}",
    category: "basic",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      const total = inputShape.reduce((a, b) => a * b, 1);
      return [total];
    }
  },
  // ========== CV - Computer Vision ==========
  conv2d: {
    type: "conv2d",
    name: "Conv2D",
    icon: "\u{1F537}",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv2dOutputShape
  },
  conv3d: {
    type: "conv3d",
    name: "Conv3D",
    icon: "\u{1F9CA}",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 4) {
        const [, d, h, w] = inputShape;
        const { outChannels } = params;
        return [outChannels, convOutLen(d, params, 0), convOutLen(h, params, 1), convOutLen(w, params, 2)];
      }
      return inputShape;
    }
  },
  depthwiseConv2d: {
    type: "depthwiseConv2d",
    name: "DepthwiseConv2D",
    icon: "\u{1F539}",
    category: "cv",
    defaultParams: { depthMultiplier: 1, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const { depthMultiplier } = params;
        const outChannels = c * depthMultiplier;
        return [outChannels, convOutLen(h, params, 0), convOutLen(w, params, 1)];
      }
      return inputShape;
    }
  },
  separableConv2d: {
    type: "separableConv2d",
    name: "SeparableConv2D",
    icon: "\u{1F536}",
    category: "cv",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv2dOutputShape
  },
  transposeConv2d: {
    type: "transposeConv2d",
    name: "TransposeConv2D",
    icon: "\u2B06\uFE0F",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 2, padding: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        const { outChannels } = params;
        const tOut = (len, i) => {
          const k = convDim(params.kernelSize, i, 3);
          const s = convDim(params.stride, i, 1);
          const p = convDim(params.padding, i, 0);
          const d = convDim(params.dilation, i, 1);
          const op = convDim(params.outputPadding, i, 0);
          return (len - 1) * s - 2 * p + d * (k - 1) + op + 1;
        };
        return [outChannels, tOut(h, 0), tOut(w, 1)];
      }
      return inputShape;
    }
  },
  maxpool2d: {
    type: "maxpool2d",
    name: "MaxPool2D",
    icon: "\u2B07\uFE0F",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2, padding: 0 },
    computeOutputShape: pool2dOutputShape
  },
  avgpool2d: {
    type: "avgpool2d",
    name: "AvgPool2D",
    icon: "\u{1F4CA}",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2, padding: 0 },
    computeOutputShape: pool2dOutputShape
  },
  adaptiveMaxPool2d: {
    type: "adaptiveMaxPool2d",
    name: "AdaptiveMaxPool2D",
    icon: "\u{1F4D0}",
    category: "cv",
    defaultParams: { outputSize: [1, 1] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        const raw = params.outputSize ?? [1, 1];
        const outputSize = Array.isArray(raw) ? raw : [raw, raw];
        return [c, ...outputSize];
      }
      return inputShape;
    }
  },
  adaptiveAvgPool2d: {
    type: "adaptiveAvgPool2d",
    name: "AdaptiveAvgPool2D",
    icon: "\u{1F4D0}",
    category: "cv",
    defaultParams: { outputSize: [1, 1] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        const raw = params.outputSize ?? [1, 1];
        const outputSize = Array.isArray(raw) ? raw : [raw, raw];
        return [c, ...outputSize];
      }
      return inputShape;
    }
  },
  globalAvgPool2d: {
    type: "globalAvgPool2d",
    name: "GlobalAvgPool2D",
    icon: "\u{1F310}",
    category: "cv",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        return [c];
      }
      return inputShape;
    }
  },
  upsample: {
    type: "upsample",
    name: "Upsample",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { scaleFactor: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const scale = params.scaleFactor || 2;
        return [c, h * scale, w * scale];
      }
      return inputShape;
    }
  },
  pixelShuffle: {
    type: "pixelShuffle",
    name: "PixelShuffle",
    icon: "\u{1F532}",
    category: "cv",
    defaultParams: { upscaleFactor: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const r = params.upscaleFactor || 2;
        return [Math.floor(c / (r * r)), h * r, w * r];
      }
      return inputShape;
    }
  },
  // ========== NLP - Natural Language Processing ==========
  conv1d: {
    type: "conv1d",
    name: "Conv1D",
    icon: "\u{1F538}",
    category: "nlp",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv1dOutputShape
  },
  maxpool1d: {
    type: "maxpool1d",
    name: "MaxPool1D",
    icon: "\u2B07\uFE0F",
    category: "nlp",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const [c, l] = inputShape;
        return [c, convOutLen(l, { ...params, stride: params.stride ?? params.kernelSize }, 0)];
      }
      return inputShape;
    }
  },
  avgpool1d: {
    type: "avgpool1d",
    name: "AvgPool1D",
    icon: "\u{1F4CA}",
    category: "nlp",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const [c, l] = inputShape;
        return [c, convOutLen(l, { ...params, stride: params.stride ?? params.kernelSize }, 0)];
      }
      return inputShape;
    }
  },
  embedding: {
    type: "embedding",
    name: "Embedding",
    icon: "\u{1F524}",
    category: "nlp",
    defaultParams: { numEmbeddings: 3e4, embeddingDim: 128 },
    // Embedding consumes integer token ids and APPENDS an embed dim:
    // [seq] -> [seq, dim], [1, seq] -> [1, seq, dim]. The old rank>=2 branch
    // replaced the last dim, silently deleting the sequence axis.
    computeOutputShape: (inputShape, params) => {
      const dim = params.embeddingDim || params.embedDim || 128;
      return [...inputShape, dim];
    }
  },
  segmentEmbedding: {
    type: "segmentEmbedding",
    name: "Segment Embedding",
    icon: "\u{1FAAA}",
    category: "nlp",
    defaultParams: { numSegments: 2, embeddingDim: 768 },
    // Adds a segment lookup to existing hidden states: shape-preserving.
    // Overwriting the last dim with embeddingDim would mask real width
    // mismatches (same bug the plain embedding/positionalEncoding rules had).
    computeOutputShape: (inputShape) => [...inputShape]
  },
  lstm: {
    type: "lstm",
    name: "LSTM",
    icon: "\u{1F504}",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: false },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, params.bidirectional ? 2 : 1)
  },
  gru: {
    type: "gru",
    name: "GRU",
    icon: "\u{1F500}",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: false },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, params.bidirectional ? 2 : 1)
  },
  rnn: {
    type: "rnn",
    name: "RNN",
    icon: "\u21A9\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, 1)
  },
  bidirectionalLSTM: {
    type: "bidirectionalLSTM",
    name: "BiLSTM",
    icon: "\u2194\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => recurrentOutputShape(inputShape, params, 2)
  },
  bidirectionalGRU: {
    type: "bidirectionalGRU",
    name: "BiGRU",
    icon: "\u2194\uFE0F",
    category: "nlp",
    defaultParams: { hiddenSize: 128, numLayers: 1 },
    computeOutputShape: (inputShape, params) => {
      const hiddenSize = params.hiddenSize || 128;
      if (inputShape.length >= 3) {
        return [inputShape[0], inputShape[1], hiddenSize * 2];
      }
      return [inputShape[0] || 1, hiddenSize * 2];
    }
  },
  attention: {
    type: "attention",
    name: "Attention",
    icon: "\u{1F441}\uFE0F",
    category: "nlp",
    defaultParams: { embedDim: 128, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  selfAttention: {
    type: "selfAttention",
    name: "Self-Attention",
    icon: "\u{1F50D}",
    category: "nlp",
    defaultParams: { embedDim: 128, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  crossAttention: {
    type: "crossAttention",
    name: "Cross-Attention",
    icon: "\u{1F500}",
    category: "nlp",
    defaultParams: { embedDim: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  globalAvgPool1d: {
    type: "globalAvgPool1d",
    name: "GlobalAvgPool1D",
    icon: "\u{1F310}",
    category: "nlp",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 2) return [inputShape[0]];
      return inputShape;
    }
  },
  depthwiseConv1d: {
    type: "depthwiseConv1d",
    name: "DepthwiseConv1D",
    icon: "\u{1F539}",
    category: "nlp",
    defaultParams: { channels: 64, kernelSize: 31, padding: 15, stride: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const [c, l] = inputShape;
        const { kernelSize, stride = 1, padding = 0 } = params;
        return [c, Math.floor((l + 2 * padding - kernelSize) / stride + 1)];
      }
      return inputShape;
    }
  },
  // ========== LLM - Large Language Models ==========
  groupedQueryAttention: {
    type: "groupedQueryAttention",
    name: "Grouped Query Attn",
    icon: "\u{1F465}",
    // flashAttention keeps attention memory O(seq) at prefill (no N×N score
    // matrix materialised); it changes runtime memory, not the math/shapes.
    category: "llm",
    defaultParams: { embedDim: 4096, numHeads: 32, numKVHeads: 8, flashAttention: true },
    computeOutputShape: (inputShape) => inputShape
  },
  causalAttention: {
    type: "causalAttention",
    name: "Causal Attention",
    icon: "\u{1F512}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  multiHeadAttention: {
    type: "multiHeadAttention",
    name: "Multi-Head Attention",
    icon: "\u{1F3AF}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, flashAttention: false },
    computeOutputShape: (inputShape) => inputShape
  },
  localAttention: {
    type: "localAttention",
    name: "Local Attention",
    icon: "\u{1FA9F}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, windowSize: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  linearAttention: {
    type: "linearAttention",
    name: "Linear Attention",
    icon: "\u{1F4C8}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, featureMap: "elu" },
    computeOutputShape: (inputShape) => inputShape
  },
  transformerBlock: {
    type: "transformerBlock",
    name: "Transformer Block",
    icon: "\u2699\uFE0F",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, ffDim: 2048 },
    computeOutputShape: (inputShape) => inputShape
  },
  positionalEncoding: {
    type: "positionalEncoding",
    name: "Positional Encoding",
    icon: "\u{1F4CD}",
    category: "llm",
    defaultParams: { maxLen: 512, embedDim: 512 },
    // Adding a positional encoding never changes the tensor shape. The old
    // behaviour overwrote the last dim with the layer's own embedDim, which
    // MASKED real width mismatches downstream instead of surfacing them
    // (the shape gate compares upstream width against attention/linear dims,
    // so passing the true input through lets that check fire).
    computeOutputShape: (inputShape) => inputShape
  },
  learnedPositionalEmbedding: {
    type: "learnedPositionalEmbedding",
    name: "Learned Pos Embed",
    icon: "\u{1F4CD}",
    category: "llm",
    defaultParams: { maxLen: 512, embedDim: 768 },
    // Added to existing hidden states, like positionalEncoding: shape-preserving.
    computeOutputShape: (inputShape) => [...inputShape]
  },
  feedForward: {
    type: "feedForward",
    name: "Feed Forward",
    icon: "\u27A1\uFE0F",
    category: "nlp",
    defaultParams: { embedDim: 512, ffDim: 2048 },
    computeOutputShape: (inputShape, params) => {
      const last = inputShape.length ? inputShape[inputShape.length - 1] : void 0;
      const out = params.embedDim ?? params.hiddenDim ?? params.dModel ?? last ?? 512;
      return inputShape.length >= 2 ? [...inputShape.slice(0, -1), out] : [out];
    }
  },
  rope: {
    type: "rope",
    name: "RoPE",
    icon: "\u{1F300}",
    category: "llm",
    // scalingType extends usable context past the training length:
    //   none        — vanilla RoPE, no extrapolation
    //   linear      — Positional Interpolation (PI): divide positions by factor
    //   dynamic-ntk — NTK-aware dynamic scaling (adjusts base by factor)
    //   yarn        — YaRN (NTK-by-parts), best long-context retention
    // originalMaxPos is the training context; effective context ≈ originalMaxPos × scalingFactor.
    defaultParams: { dim: 128, scalingType: "none", scalingFactor: 1, originalMaxPos: 2048 },
    computeOutputShape: (inputShape) => inputShape
  },
  rmsNorm: {
    type: "rmsNorm",
    name: "RMSNorm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: { normalizedShape: 512 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Audio ==========
  melSpectrogram: {
    type: "melSpectrogram",
    name: "Mel Spectrogram",
    icon: "\u{1F3B5}",
    category: "audio",
    defaultParams: { nMelBands: 80, hopLength: 512 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.nMelBands || 80, Math.floor(inputShape[1] / (params.hopLength || 512))];
      }
      return [params.nMelBands || 80, 128];
    }
  },
  mfcc: {
    type: "mfcc",
    name: "MFCC",
    icon: "\u{1F3BC}",
    category: "audio",
    defaultParams: { nMFCC: 13 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.nMFCC || 13, Math.floor(inputShape[1] / 512)];
      }
      return [params.nMFCC || 13, 128];
    }
  },
  stft: {
    type: "stft",
    name: "STFT",
    icon: "\u{1F4FB}",
    category: "audio",
    defaultParams: { nFFT: 2048, hopLength: 512 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const nFFT = params.nFFT || 2048;
        const timeFrames = Math.floor(inputShape[1] / (params.hopLength || 512));
        return [inputShape[0], Math.floor(nFFT / 2) + 1, timeFrames];
      }
      return [1025, 128];
    }
  },
  audioConv: {
    type: "audioConv",
    name: "Audio Conv",
    icon: "\u{1F50A}",
    category: "audio",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1 },
    computeOutputShape: conv1dOutputShape
  },
  conformerBlock: {
    type: "conformerBlock",
    name: "Conformer Block",
    icon: "\u{1F399}\uFE0F",
    category: "audio",
    defaultParams: { dModel: 256, numHeads: 4, ffMult: 4, kernelSize: 31 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Tabular ==========
  featureInteraction: {
    type: "featureInteraction",
    name: "Feature Interaction",
    icon: "\u{1F517}",
    category: "tabular",
    defaultParams: { interactionDim: 64 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.interactionDim || 64];
    }
  },
  embeddingBag: {
    type: "embeddingBag",
    name: "EmbeddingBag",
    icon: "\u{1F392}",
    category: "tabular",
    defaultParams: { vocabSize: 1e3, embedDim: 32 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.embedDim || 32];
    }
  },
  tabnet: {
    type: "tabnet",
    name: "TabNet",
    icon: "\u{1F4CB}",
    category: "tabular",
    defaultParams: { featureDim: 64, decisionDim: 64 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.decisionDim || 64];
    }
  },
  // ========== Reinforcement Learning ==========
  dqnHead: {
    type: "dqnHead",
    name: "DQN Head",
    icon: "\u{1F3AE}",
    category: "rl",
    defaultParams: { numActions: 4 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.numActions || 4];
    }
  },
  actorHead: {
    type: "actorHead",
    name: "Actor Head",
    icon: "\u{1F3AD}",
    category: "rl",
    defaultParams: { numActions: 4 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.numActions || 4];
    }
  },
  criticHead: {
    type: "criticHead",
    name: "Critic Head",
    icon: "\u2B50",
    category: "rl",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      return [inputShape[0] || 1, 1];
    }
  },
  policyNetwork: {
    type: "policyNetwork",
    name: "Policy Network",
    icon: "\u{1F4DC}",
    category: "rl",
    defaultParams: { numActions: 4 },
    computeOutputShape: (inputShape, params) => {
      return [inputShape[0] || 1, params.numActions || 4];
    }
  },
  valueNetwork: {
    type: "valueNetwork",
    name: "Value Network",
    icon: "\u{1F4B0}",
    category: "rl",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      return [inputShape[0] || 1, 1];
    }
  },
  // ========== Graph ML ==========
  graphConv: {
    type: "graphConv",
    name: "GraphConv",
    icon: "\u{1F578}\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  graphAttention: {
    type: "graphAttention",
    name: "Graph Attention",
    icon: "\u{1F577}\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64, numHeads: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  graphSAGE: {
    type: "graphSAGE",
    name: "GraphSAGE",
    icon: "\u{1F4CA}",
    category: "graph",
    defaultParams: { outFeatures: 64 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  gcn: {
    type: "gcn",
    name: "GCN",
    icon: "\u{1F517}",
    category: "graph",
    defaultParams: { outFeatures: 64 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [inputShape[0], params.outFeatures || 64];
      }
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  gin: {
    type: "gin",
    name: "GIN",
    icon: "\u{1F52E}",
    category: "graph",
    defaultParams: { outFeatures: 64, eps: 0 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) return [inputShape[0], params.outFeatures || 64];
      return inputShape;
    }
  },
  gat: {
    type: "gat",
    name: "GAT",
    icon: "\u{1F441}\uFE0F\u200D\u{1F5E8}\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64, numHeads: 8 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        const outDim = params.outFeatures || 64;
        const heads = params.numHeads || 8;
        return [inputShape[0], outDim * heads];
      }
      return [inputShape[0] || 1, (params.outFeatures || 64) * (params.numHeads || 8)];
    }
  },
  edgeConv: {
    type: "edgeConv",
    name: "EdgeConv",
    icon: "\u2702\uFE0F",
    category: "graph",
    defaultParams: { outFeatures: 64, k: 20 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) return [inputShape[0], params.outFeatures || 64];
      return [inputShape[0] || 1, params.outFeatures || 64];
    }
  },
  // ========== Multimodal ==========
  crossModalAttention: {
    type: "crossModalAttention",
    name: "Cross-Modal Attention",
    icon: "\u{1F500}",
    category: "multimodal",
    defaultParams: { embedDim: 256, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  fusion: {
    type: "fusion",
    name: "Fusion",
    icon: "\u{1F501}",
    category: "multimodal",
    defaultParams: { fusionDim: 256, method: "concat" },
    computeOutputShape: (inputShape, params) => {
      if (params.method === "concat") {
        return [inputShape[0] || 1, params.fusionDim || 256];
      }
      return inputShape;
    }
  },
  projection: {
    type: "projection",
    name: "Projection",
    icon: "\u27A1\uFE0F",
    category: "multimodal",
    defaultParams: { outDim: 256 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [...inputShape.slice(0, -1), params.outDim || 256];
      }
      return [params.outDim || 256];
    }
  },
  coAttention: {
    type: "coAttention",
    name: "Co-Attention",
    icon: "\u{1F501}",
    category: "multimodal",
    defaultParams: { embedDim: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Activation ==========
  relu: {
    type: "relu",
    name: "ReLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  relu6: {
    type: "relu6",
    name: "ReLU6",
    icon: "\u26A1",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  softplus: {
    type: "softplus",
    name: "Softplus",
    icon: "\u301C",
    category: "activation",
    defaultParams: { beta: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  leakyRelu: {
    type: "leakyRelu",
    name: "LeakyReLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { negativeSlope: 0.01 },
    computeOutputShape: (inputShape) => inputShape
  },
  elu: {
    type: "elu",
    name: "ELU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { alpha: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  prelu: {
    type: "prelu",
    name: "PReLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { numParameters: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  gelu: {
    type: "gelu",
    name: "GELU",
    icon: "\u2728",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  swish: {
    type: "swish",
    name: "Swish",
    icon: "\u{1F4AB}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  sigmoid: {
    type: "sigmoid",
    name: "Sigmoid",
    icon: "\u{1F4C8}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  tanh: {
    type: "tanh",
    name: "Tanh",
    icon: "\u{1F30A}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  softmax: {
    type: "softmax",
    name: "Softmax",
    icon: "\u{1F4CA}",
    category: "activation",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape) => inputShape
  },
  gumbelSoftmax: {
    type: "gumbelSoftmax",
    name: "Gumbel-Softmax",
    icon: "\u{1F3B2}",
    category: "activation",
    defaultParams: { tau: 1, hard: false, dim: -1 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== Activation extras ==========
  selu: {
    type: "selu",
    name: "SELU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  mish: {
    type: "mish",
    name: "Mish",
    icon: "\u301C",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  hardSwish: {
    type: "hardSwish",
    name: "HardSwish",
    icon: "\u2312",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  hardSigmoid: {
    type: "hardSigmoid",
    name: "HardSigmoid",
    icon: "\u{1F4C8}",
    category: "activation",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  logSoftmax: {
    type: "logSoftmax",
    name: "LogSoftmax",
    icon: "\u{1F4C9}",
    category: "activation",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape) => inputShape
  },
  glu: {
    type: "glu",
    name: "GLU",
    icon: "\u{1F6AA}",
    category: "activation",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      const out = [...inputShape];
      out[idx] = Math.floor(inputShape[idx] / 2);
      return out;
    }
  },
  // ========== Normalization ==========
  batchNorm: {
    type: "batchNorm",
    name: "BatchNorm",
    icon: "\u{1F4CF}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  layerNorm: {
    type: "layerNorm",
    name: "LayerNorm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: { normalizedShape: [512] },
    computeOutputShape: (inputShape) => inputShape
  },
  instanceNorm: {
    type: "instanceNorm",
    name: "InstanceNorm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  groupNorm: {
    type: "groupNorm",
    name: "GroupNorm",
    icon: "\u{1F4D1}",
    category: "normalization",
    defaultParams: { numGroups: 32 },
    computeOutputShape: (inputShape) => inputShape
  },
  adaIN: {
    type: "adaIN",
    name: "AdaIN",
    icon: "\u{1F3A8}",
    category: "normalization",
    defaultParams: { numFeatures: 512 },
    computeOutputShape: (inputShape) => inputShape
  },
  spectralNorm: {
    type: "spectralNorm",
    name: "SpectralNorm",
    icon: "\u{1F4E1}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  weightNorm: {
    type: "weightNorm",
    name: "WeightNorm",
    icon: "\u2696\uFE0F",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  localResponseNorm: {
    type: "localResponseNorm",
    name: "LocalResponseNorm",
    icon: "\u{1F4CA}",
    category: "normalization",
    defaultParams: { size: 5, alpha: 1e-4, beta: 0.75, k: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  pixelNorm: {
    type: "pixelNorm",
    name: "PixelNorm",
    icon: "\u{1F3A8}",
    category: "normalization",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== LLM extras ==========
  swiglu: {
    type: "swiglu",
    name: "SwiGLU",
    icon: "\u26A1",
    category: "llm",
    defaultParams: { embedDim: 4096, intermediateSize: 11008 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 1) {
        return [...inputShape.slice(0, -1), params.embedDim ?? inputShape[inputShape.length - 1]];
      }
      return inputShape;
    }
  },
  moeLayer: {
    type: "moeLayer",
    name: "MoE Layer",
    icon: "\u{1F500}",
    category: "llm",
    defaultParams: { embedDim: 4096, numExperts: 8, topK: 2, expertDim: 14336 },
    // Shape passes through, but a router asked for more experts than exist is
    // a guaranteed config bug worth blocking at design time.
    computeOutputShape: (inputShape, params) => {
      const k = Number(params.topK ?? 2);
      const e = Number(params.numExperts ?? 8);
      if (Number.isFinite(k) && Number.isFinite(e) && k > e) {
        throw new Error(`moeLayer: topK (${k}) exceeds numExperts (${e})`);
      }
      return inputShape;
    }
  },
  alibi: {
    type: "alibi",
    name: "ALiBi",
    icon: "\u{1F4D0}",
    category: "llm",
    defaultParams: { numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  lmHead: {
    type: "lmHead",
    name: "LM Head",
    icon: "\u{1F5E3}\uFE0F",
    category: "llm",
    defaultParams: { vocabSize: 50257 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 2) {
        return [...inputShape.slice(0, -1), params.vocabSize || 50257];
      }
      return [params.vocabSize || 50257];
    }
  },
  timeEmbedding: {
    type: "timeEmbedding",
    name: "Time Embedding",
    icon: "\u23F1\uFE0F",
    category: "llm",
    defaultParams: { dim: 256 },
    computeOutputShape: (_inputShape, params) => [params.dim || 256]
  },
  mamba: {
    type: "mamba",
    name: "Mamba (SSM)",
    icon: "\u{1F40D}",
    category: "llm",
    defaultParams: { dModel: 256, dState: 16, dConv: 4, expand: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  relativePositionBias: {
    type: "relativePositionBias",
    name: "Relative Position Bias",
    icon: "\u{1F4CD}",
    category: "llm",
    defaultParams: { numHeads: 8, numBuckets: 32, maxDistance: 128 },
    computeOutputShape: (inputShape) => inputShape
  },
  // ========== CV extras ==========
  fpn: {
    type: "fpn",
    name: "FPN",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { inChannels: [256, 512, 1024, 2048], outChannels: 256 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        return [params.outChannels || 256, inputShape[1], inputShape[2]];
      }
      return [params.outChannels || 256];
    }
  },
  invResidualBlock: {
    type: "invResidualBlock",
    name: "Inv. Residual (MBConv)",
    icon: "\u{1F4F1}",
    category: "cv",
    defaultParams: { inChannels: 32, outChannels: 32, expandRatio: 6, stride: 1 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        const stride = params.stride || 1;
        return [params.outChannels || inputShape[0], Math.floor(h / stride), Math.floor(w / stride)];
      }
      return inputShape;
    }
  },
  deformableConv2d: {
    type: "deformableConv2d",
    name: "DeformableConv2D",
    icon: "\u{1F300}",
    category: "cv",
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
    computeOutputShape: conv2dOutputShape
  },
  interpolate: {
    type: "interpolate",
    name: "Interpolate",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { scaleFactor: 2, mode: "bilinear" },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        if (params.size) {
          const size = Array.isArray(params.size) ? params.size : [params.size, params.size];
          return [c, size[0], size[1]];
        }
        const scale = params.scaleFactor || 2;
        return [c, Math.floor(h * scale), Math.floor(w * scale)];
      }
      return inputShape;
    }
  },
  channelShuffle: {
    type: "channelShuffle",
    name: "Channel Shuffle",
    icon: "\u{1F500}",
    category: "cv",
    defaultParams: { groups: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  gridSample: {
    type: "gridSample",
    name: "Grid Sample",
    icon: "\u{1F5FA}\uFE0F",
    category: "cv",
    defaultParams: { mode: "bilinear", paddingMode: "zeros", alignCorners: false },
    computeOutputShape: (inputShape) => inputShape
  },
  spatialPyramidPool: {
    type: "spatialPyramidPool",
    name: "Spatial Pyramid Pool",
    icon: "\u{1F53A}",
    category: "cv",
    defaultParams: { levels: [1, 2, 4] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const c = inputShape[0];
        const levels = params.levels || [1, 2, 4];
        const total = levels.reduce((acc, l) => acc + l * l, 0);
        return [c * total];
      }
      return inputShape;
    }
  },
  dilatedConv2d: {
    type: "dilatedConv2d",
    name: "DilatedConv2D",
    icon: "\u{1F537}",
    category: "cv",
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 2, dilation: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        const { outChannels } = params;
        return [outChannels, convOutLen(h, params, 0), convOutLen(w, params, 1)];
      }
      return inputShape;
    }
  },
  globalMaxPool2d: {
    type: "globalMaxPool2d",
    name: "GlobalMaxPool2D",
    icon: "\u{1F310}",
    category: "cv",
    defaultParams: {},
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 3) {
        return [inputShape[0]];
      }
      return inputShape;
    }
  },
  roiAlign: {
    type: "roiAlign",
    name: "RoIAlign",
    icon: "\u{1F3AF}",
    category: "cv",
    defaultParams: { outputSize: [7, 7], spatialScale: 0.25 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c] = inputShape;
        const size = Array.isArray(params.outputSize) ? params.outputSize : [params.outputSize, params.outputSize];
        return [c, size[0] ?? 7, size[1] ?? 7];
      }
      return inputShape;
    }
  },
  maxpool3d: {
    type: "maxpool3d",
    name: "MaxPool3D",
    icon: "\u2B07\uFE0F",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 4) {
        const [c, d, h, w] = inputShape;
        const { kernelSize, stride = kernelSize } = params;
        return [c, Math.floor((d - kernelSize) / stride + 1), Math.floor((h - kernelSize) / stride + 1), Math.floor((w - kernelSize) / stride + 1)];
      }
      return inputShape;
    }
  },
  avgpool3d: {
    type: "avgpool3d",
    name: "AvgPool3D",
    icon: "\u{1F4CA}",
    category: "cv",
    defaultParams: { kernelSize: 2, stride: 2 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 4) {
        const [c, d, h, w] = inputShape;
        const { kernelSize, stride = kernelSize } = params;
        return [c, Math.floor((d - kernelSize) / stride + 1), Math.floor((h - kernelSize) / stride + 1), Math.floor((w - kernelSize) / stride + 1)];
      }
      return inputShape;
    }
  },
  windowAttention: {
    type: "windowAttention",
    name: "Window Attention",
    icon: "\u{1FA9F}",
    category: "cv",
    defaultParams: { embedDim: 96, numHeads: 3, windowSize: 7 },
    computeOutputShape: (inputShape) => inputShape
  },
  seBlock: {
    type: "seBlock",
    name: "SE Block",
    icon: "\u{1F3AF}",
    category: "cv",
    defaultParams: { channels: 64, reductionRatio: 16 },
    computeOutputShape: (inputShape) => inputShape
  },
  patchEmbed: {
    type: "patchEmbed",
    name: "Patch Embed",
    icon: "\u{1F532}",
    category: "cv",
    defaultParams: { imgSize: 224, patchSize: 16, embedDim: 768 },
    // Patch count comes from the ACTUAL upstream feature map when there is one,
    // and only falls back to the declared imgSize when this is the first layer
    // (rank < 3 input, or no parent). Reading imgSize unconditionally produced
    // NaN for every patchEmbed that omits it (the seeded HYB family and the
    // patch-tst template both do), and NaN silently poisoned every downstream
    // shape. Non-square inputs are handled per-axis rather than squared.
    computeOutputShape: (inputShape, params) => {
      const patch = params.patchSize || 16;
      const embedDim = params.embedDim ?? 768;
      if (inputShape.length >= 3) {
        const [, h, w] = inputShape;
        return [Math.floor(h / patch) * Math.floor(w / patch), embedDim];
      }
      const side = Math.floor((params.imgSize ?? 224) / patch);
      return [side * side, embedDim];
    }
  },
  // ========== Utility ==========
  reshape: {
    type: "reshape",
    name: "Reshape",
    icon: "\u{1F504}",
    category: "utility",
    defaultParams: { shape: [512] },
    computeOutputShape: (_inputShape, params) => {
      return Array.isArray(params.shape) ? params.shape : [params.shape || 512];
    }
  },
  dropout: {
    type: "dropout",
    name: "Dropout",
    icon: "\u{1F3B2}",
    category: "utility",
    defaultParams: { p: 0.5 },
    computeOutputShape: (inputShape) => inputShape
  },
  residual: {
    type: "residual",
    name: "Residual",
    icon: "\u2795",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  skipConnection: {
    type: "skipConnection",
    name: "Skip Connection",
    icon: "\u23ED\uFE0F",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  concatenate: {
    type: "concatenate",
    name: "Concatenate",
    icon: "\u{1F517}",
    category: "utility",
    defaultParams: { dim: -1 },
    computeOutputShape: (inputShape, params, allInputShapes) => {
      const shapes = allInputShapes && allInputShapes.length > 1 ? allInputShapes : [inputShape, inputShape];
      const rank = shapes[0].length;
      const dim = params.dim === -1 ? rank - 1 : params.dim ?? rank - 1;
      const concatDimTotal = shapes.reduce((sum, s) => sum + (s[dim] ?? 0), 0);
      const out = [...shapes[0]];
      out[dim] = concatDimTotal;
      return out;
    }
  },
  add: {
    type: "add",
    name: "Add",
    icon: "\u2795",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  multiply: {
    type: "multiply",
    name: "Multiply",
    icon: "\u2716\uFE0F",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape) => inputShape
  },
  dropPath: {
    type: "dropPath",
    name: "DropPath",
    icon: "\u{1FA82}",
    category: "utility",
    defaultParams: { dropRate: 0.1 },
    computeOutputShape: (inputShape) => inputShape
  },
  layerScale: {
    type: "layerScale",
    name: "LayerScale",
    icon: "\u2696\uFE0F",
    category: "utility",
    defaultParams: { dim: 512, initValues: 1e-5 },
    computeOutputShape: (inputShape) => inputShape
  },
  split: {
    type: "split",
    name: "Split",
    icon: "\u2702\uFE0F",
    category: "utility",
    defaultParams: { sections: 2, dim: -1 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const sections = params.sections ?? 2;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (idx >= 0 && idx < inputShape.length) {
        const out = [...inputShape];
        out[idx] = Math.floor(inputShape[idx] / sections);
        return out;
      }
      return inputShape;
    }
  },
  permute: {
    type: "permute",
    name: "Permute",
    icon: "\u2194\uFE0F",
    category: "utility",
    defaultParams: { dims: [0, 2, 1] },
    computeOutputShape: (inputShape, params) => {
      const dims = params.dims ?? [0, 2, 1];
      if (dims.length === inputShape.length) {
        return dims.map((d) => inputShape[d]);
      }
      return inputShape;
    }
  },
  customModule: {
    type: "customModule",
    name: "Custom Module",
    icon: "\u{1F9E9}",
    category: "utility",
    defaultParams: { _customLayerId: "" },
    computeOutputShape: (inputShape) => inputShape
  },
  stickyNote: {
    type: "stickyNote",
    name: "Sticky Note",
    icon: "\u{1F4DD}",
    category: "utility",
    defaultParams: { _noteText: "", _noteColor: "#fef08a" },
    computeOutputShape: (inputShape) => inputShape
  },
  squeeze: {
    type: "squeeze",
    name: "Squeeze",
    icon: "\u2195\uFE0F",
    category: "utility",
    defaultParams: { dim: null },
    computeOutputShape: (inputShape, params) => {
      if (params.dim !== null && params.dim !== void 0) {
        const idx = params.dim < 0 ? inputShape.length + params.dim : params.dim;
        if (inputShape[idx] === 1) {
          return [...inputShape.slice(0, idx), ...inputShape.slice(idx + 1)];
        }
        return inputShape;
      }
      return inputShape.filter((d) => d !== 1);
    }
  },
  unsqueeze: {
    type: "unsqueeze",
    name: "Unsqueeze",
    icon: "\u2194\uFE0F",
    category: "utility",
    defaultParams: { dim: 0 },
    computeOutputShape: (inputShape, params) => {
      const idx = (params.dim ?? 0) < 0 ? inputShape.length + 1 + params.dim : params.dim ?? 0;
      const out = [...inputShape];
      out.splice(idx, 0, 1);
      return out;
    }
  },
  pad: {
    type: "pad",
    name: "Pad",
    icon: "\u{1F532}",
    category: "utility",
    defaultParams: { padding: [0, 0, 0, 0] },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 3) {
        const [c, h, w] = inputShape;
        const p = params.padding || [0, 0, 0, 0];
        return [c, h + (p[2] ?? 0) + (p[3] ?? 0), w + (p[0] ?? 0) + (p[1] ?? 0)];
      }
      return inputShape;
    }
  },
  mean: {
    type: "mean",
    name: "Mean",
    icon: "\u2797",
    category: "utility",
    defaultParams: { dim: -1, keepdim: false },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (params.keepdim) {
        const out = [...inputShape];
        out[idx] = 1;
        return out;
      }
      return [...inputShape.slice(0, idx), ...inputShape.slice(idx + 1)];
    }
  },
  matmul: {
    type: "matmul",
    name: "MatMul",
    icon: "\u2716\uFE0F",
    category: "utility",
    defaultParams: {},
    computeOutputShape: (inputShape, _params, allInputShapes) => {
      const shapes = allInputShapes && allInputShapes.length >= 2 ? allInputShapes : [inputShape, inputShape];
      const a = shapes[0];
      const b = shapes[1];
      if (a.length >= 2 && b.length >= 2) {
        return [...a.slice(0, -1), b[b.length - 1]];
      }
      return inputShape;
    }
  },
  clamp: {
    type: "clamp",
    name: "Clamp",
    icon: "\u{1F4CC}",
    category: "utility",
    defaultParams: { min: 0, max: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  norm: {
    type: "norm",
    name: "L2 Norm",
    icon: "\u{1F4CF}",
    category: "utility",
    defaultParams: { dim: -1, p: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  vaeBottleneck: {
    type: "vaeBottleneck",
    name: "VAE Bottleneck",
    icon: "\u{1F9EC}",
    category: "utility",
    defaultParams: { latentDim: 128 },
    computeOutputShape: (_inputShape, params) => [params.latentDim || 128]
  },
  miniBatchStdDev: {
    type: "miniBatchStdDev",
    name: "MiniBatch StdDev",
    icon: "\u{1F4CA}",
    category: "utility",
    defaultParams: { groupSize: 4 },
    computeOutputShape: (inputShape) => {
      if (inputShape.length >= 3) {
        return [inputShape[0] + 1, inputShape[1], inputShape[2]];
      }
      return inputShape;
    }
  },
  topK: {
    type: "topK",
    name: "TopK",
    icon: "\u{1F51D}",
    category: "utility",
    defaultParams: { k: 2, dim: -1, sorted: true },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? -1;
      const idx = dim < 0 ? inputShape.length + dim : dim;
      const out = [...inputShape];
      out[idx] = params.k ?? 2;
      return out;
    }
  },
  gather: {
    type: "gather",
    name: "Gather",
    icon: "\u{1FA9D}",
    category: "utility",
    // indexSize 0 = "same as input" (index tensor length unknown at design
    // time); a positive value resizes the gathered dim to the index length.
    defaultParams: { dim: 0, indexSize: 0 },
    computeOutputShape: (inputShape, params) => {
      const dim = Number(params.dim ?? 0);
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (idx < 0 || idx >= inputShape.length) {
        throw new Error(`gather: dim ${params.dim} out of range for rank ${inputShape.length}`);
      }
      const out = [...inputShape];
      const n = Number(params.indexSize);
      if (Number.isFinite(n) && n > 0) out[idx] = n;
      return out;
    }
  },
  scatter: {
    type: "scatter",
    name: "Scatter",
    icon: "\u{1F4A7}",
    category: "utility",
    defaultParams: { dim: 0 },
    // Output shape IS the self tensor's shape (passthrough is correct); the
    // only design-time checkable bug is an out-of-range dim.
    computeOutputShape: (inputShape, params) => {
      const dim = Number(params.dim ?? 0);
      const idx = dim < 0 ? inputShape.length + dim : dim;
      if (idx < 0 || idx >= inputShape.length) {
        throw new Error(`scatter: dim ${params.dim} out of range for rank ${inputShape.length}`);
      }
      return inputShape;
    }
  },
  stack: {
    type: "stack",
    name: "Stack",
    icon: "\u{1F4DA}",
    category: "utility",
    defaultParams: { dim: 0 },
    computeOutputShape: (inputShape, params) => {
      const dim = params.dim ?? 0;
      const idx = dim < 0 ? inputShape.length + dim + 1 : dim;
      const out = [...inputShape];
      out.splice(idx, 0, 1);
      return out;
    }
  },
  einsum: {
    type: "einsum",
    name: "Einsum",
    icon: "\u2211",
    category: "utility",
    defaultParams: { equation: "ij,jk->ik" },
    // Real equation-driven shape: the first operand's subscripts bind to the
    // incoming shape (rank must agree — that mismatch is THE einsum bug worth
    // catching before runtime). Output dims resolve from those bindings;
    // letters bound only by other operands are unknowable from one input, so
    // the shape stays passthrough in that case rather than guessing.
    computeOutputShape: (inputShape, params) => {
      const eq = String(params.equation ?? "").replace(/\s+/g, "");
      const m = eq.match(/^([a-zA-Z]+(?:,[a-zA-Z]+)*)->([a-zA-Z]*)$/);
      if (!m) return inputShape;
      const first = m[1].split(",")[0];
      const rhs = m[2];
      if (first.length !== inputShape.length) {
        throw new Error(`einsum: first operand '${first}' expects rank ${first.length}, input has rank ${inputShape.length} (equation "${eq}")`);
      }
      const bound = /* @__PURE__ */ new Map();
      [...first].forEach((ch, i) => bound.set(ch, inputShape[i]));
      if (![...rhs].every((ch) => bound.has(ch))) return inputShape;
      return [...rhs].map((ch) => bound.get(ch));
    }
  },
  // ========== Frontier architectures (2024-2025) ==========
  mla: {
    type: "mla",
    name: "Multi-Head Latent Attention",
    icon: "\u{1F9EC}",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, kvLatentDim: 128, ropeHeadDim: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  mamba2: {
    type: "mamba2",
    name: "Mamba-2 (SSD)",
    icon: "\u{1F40D}",
    category: "llm",
    defaultParams: { dModel: 512, dState: 128, expand: 2, headDim: 64, chunkSize: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  qkNorm: {
    type: "qkNorm",
    name: "QK-Norm",
    icon: "\u{1F4D0}",
    category: "normalization",
    defaultParams: { dim: 64, eps: 1e-6 },
    computeOutputShape: (inputShape) => inputShape
  },
  multiTokenPrediction: {
    type: "multiTokenPrediction",
    name: "Multi-Token Prediction Head",
    icon: "\u{1F3B0}",
    category: "llm",
    defaultParams: { vocabSize: 32e3, numFutureTokens: 2, dModel: 512 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 1) {
        return [...inputShape.slice(0, -1), params.vocabSize || 32e3];
      }
      return [params.vocabSize || 32e3];
    }
  },
  xlstm: {
    type: "xlstm",
    name: "xLSTM",
    icon: "\u{1F501}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 4, variant: "mLSTM" },
    computeOutputShape: (inputShape) => inputShape
  },
  differentialAttention: {
    type: "differentialAttention",
    name: "Differential Attention",
    icon: "\u2796",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, lambdaInit: 0.8 },
    computeOutputShape: (inputShape) => inputShape
  },
  rgLru: {
    type: "rgLru",
    name: "RG-LRU (Griffin)",
    icon: "\u{1F985}",
    category: "llm",
    defaultParams: { dModel: 512, expand: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  retention: {
    type: "retention",
    name: "Retention (RetNet)",
    icon: "\u{1F9F2}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  hyena: {
    type: "hyena",
    name: "Hyena",
    icon: "\u{1F300}",
    category: "llm",
    defaultParams: { dModel: 512, order: 2, filterOrder: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  rwkv: {
    type: "rwkv",
    name: "RWKV (Time-Mix)",
    icon: "\u23F3",
    category: "llm",
    defaultParams: { dModel: 512 },
    computeOutputShape: (inputShape) => inputShape
  },
  kan: {
    type: "kan",
    name: "KAN Layer",
    icon: "\u{1FAA2}",
    category: "basic",
    defaultParams: { inFeatures: 128, outFeatures: 128, gridSize: 5, splineOrder: 3 },
    computeOutputShape: (inputShape, params) => {
      return [...inputShape.slice(0, -1), params.outFeatures || 128];
    }
  },
  mixtureOfDepths: {
    type: "mixtureOfDepths",
    name: "Mixture-of-Depths",
    icon: "\u{1FA9C}",
    category: "llm",
    defaultParams: { dModel: 512, capacityFactor: 0.5 },
    computeOutputShape: (inputShape) => inputShape
  },
  tttLayer: {
    type: "tttLayer",
    name: "Test-Time Training Layer",
    icon: "\u{1F9EA}",
    category: "llm",
    defaultParams: { dModel: 512, innerSteps: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  geglu: {
    type: "geglu",
    name: "GeGLU",
    icon: "\u26A1",
    category: "activation",
    defaultParams: { dim: 512, hiddenDim: 2048 },
    computeOutputShape: (inputShape, params) => {
      if (inputShape.length >= 1) {
        return [...inputShape.slice(0, -1), params.dim ?? inputShape[inputShape.length - 1]];
      }
      return inputShape;
    }
  },
  grn: {
    type: "grn",
    name: "Global Response Norm",
    icon: "\u{1F310}",
    category: "normalization",
    defaultParams: { channels: 256, eps: 1e-6 },
    computeOutputShape: (inputShape) => inputShape
  },
  titansMemory: {
    type: "titansMemory",
    name: "Titans Neural Memory",
    icon: "\u{1F5FF}",
    category: "llm",
    defaultParams: { dModel: 512, memoryDepth: 2 },
    computeOutputShape: (inputShape) => inputShape
  },
  deltaNet: {
    type: "deltaNet",
    name: "DeltaNet",
    icon: "\u{1F53A}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8 },
    computeOutputShape: (inputShape) => inputShape
  },
  gatedDeltaNet: {
    type: "gatedDeltaNet",
    name: "Gated DeltaNet",
    icon: "\u{1F53B}",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8, headDim: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  sharedExpertMoE: {
    type: "sharedExpertMoE",
    name: "Shared-Expert MoE",
    icon: "\u{1F9E9}",
    category: "llm",
    defaultParams: { embedDim: 4096, numExperts: 64, numSharedExperts: 2, topK: 6, expertDim: 1408 },
    computeOutputShape: (inputShape) => inputShape
  },
  ditBlock: {
    type: "ditBlock",
    name: "DiT Block (AdaLN-Zero)",
    icon: "\u{1F3A8}",
    category: "cv",
    defaultParams: { hiddenDim: 1152, numHeads: 16, condDim: 1152 },
    computeOutputShape: (inputShape) => inputShape
  },
  vectorQuantizer: {
    type: "vectorQuantizer",
    name: "Vector Quantizer (VQ)",
    icon: "\u{1F48E}",
    category: "utility",
    defaultParams: { codebookSize: 8192, embedDim: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  perceiverLatent: {
    type: "perceiverLatent",
    name: "Perceiver Latent / Q-Former",
    icon: "\u{1F52E}",
    category: "multimodal",
    defaultParams: { numLatents: 64, latentDim: 768, numHeads: 8 },
    computeOutputShape: (inputShape, params) => {
      const numLatents = params.numLatents ?? 64;
      const latentDim = params.latentDim ?? 768;
      if (inputShape.length >= 2) return [...inputShape.slice(0, -2), numLatents, latentDim];
      return [numLatents, latentDim];
    }
  },
  convNeXtBlock: {
    type: "convNeXtBlock",
    name: "ConvNeXt Block",
    icon: "\u{1F9F1}",
    category: "cv",
    defaultParams: { dim: 96, kernelSize: 7, expandRatio: 4 },
    computeOutputShape: (inputShape) => inputShape
  },
  gatedLinearAttention: {
    type: "gatedLinearAttention",
    name: "Gated Linear Attention (GLA)",
    icon: "\u{1F32C}\uFE0F",
    category: "llm",
    defaultParams: { dModel: 512, numHeads: 8, expandK: 0.5, expandV: 1 },
    computeOutputShape: (inputShape) => inputShape
  },
  s4Layer: {
    type: "s4Layer",
    name: "S4 / S5 (Structured SSM)",
    icon: "\u3030\uFE0F",
    category: "llm",
    defaultParams: { dModel: 512, dState: 64 },
    computeOutputShape: (inputShape) => inputShape
  },
  dyt: {
    type: "dyt",
    name: "Dynamic Tanh (DyT)",
    icon: "\u{1F4C9}",
    category: "normalization",
    defaultParams: { dim: 512, alphaInit: 0.5 },
    computeOutputShape: (inputShape) => inputShape
  },
  nativeSparseAttention: {
    type: "nativeSparseAttention",
    name: "Native Sparse Attention",
    icon: "\u{1F578}\uFE0F",
    category: "llm",
    defaultParams: { embedDim: 512, numHeads: 8, blockSize: 64, topBlocks: 16 },
    computeOutputShape: (inputShape) => inputShape
  },
  film: {
    type: "film",
    name: "FiLM",
    icon: "\u{1F39A}\uFE0F",
    category: "utility",
    defaultParams: { numFeatures: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  residualVQ: {
    type: "residualVQ",
    name: "Residual VQ (RVQ)",
    icon: "\u{1F4A0}",
    category: "audio",
    defaultParams: { numQuantizers: 8, codebookSize: 1024, embedDim: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  crossNetworkDCN: {
    type: "crossNetworkDCN",
    name: "DCN Cross Network",
    icon: "\u{1F517}",
    category: "tabular",
    defaultParams: { numLayers: 3, inputDim: 256 },
    computeOutputShape: (inputShape) => inputShape
  },
  ftTransformerBlock: {
    type: "ftTransformerBlock",
    name: "FT-Transformer",
    icon: "\u{1F4CB}",
    category: "tabular",
    defaultParams: { dModel: 192, numHeads: 8, ffMult: 4 },
    computeOutputShape: (inputShape) => inputShape
  },
  deformableAttention: {
    type: "deformableAttention",
    name: "Deformable Attention",
    icon: "\u{1F9ED}",
    category: "cv",
    defaultParams: { embedDim: 256, numHeads: 8, numPoints: 4, numLevels: 4 },
    computeOutputShape: (inputShape) => inputShape
  },
  attentionPool: {
    type: "attentionPool",
    name: "Attention Pooling (PMA)",
    icon: "\u{1F3AF}",
    category: "basic",
    defaultParams: { dim: 512, numHeads: 8, numSeeds: 1 },
    computeOutputShape: (inputShape, params) => {
      const numSeeds = params.numSeeds ?? 1;
      if (inputShape.length >= 2) {
        const lastDim = inputShape[inputShape.length - 1];
        if (numSeeds === 1) return [...inputShape.slice(0, -2), lastDim];
        return [...inputShape.slice(0, -2), numSeeds, lastDim];
      }
      return inputShape;
    }
  },
  // ========== Time-series / 3D / Video (2024-2025) ==========
  revIN: {
    type: "revIN",
    name: "Reversible Instance Norm",
    icon: "\u{1F501}",
    category: "normalization",
    defaultParams: { numFeatures: 7, eps: 1e-5, affine: true },
    computeOutputShape: (inputShape) => inputShape
  },
  seriesDecomp: {
    type: "seriesDecomp",
    name: "Series Decomposition",
    icon: "\u303D\uFE0F",
    category: "nlp",
    defaultParams: { kernelSize: 25 },
    // Autoformer decomposition emits seasonal + trend, BOTH input-shaped, so
    // passthrough is the true per-branch shape. The moving average needs an
    // odd kernel for symmetric padding to preserve length.
    computeOutputShape: (inputShape, params) => {
      const k = Number(params.kernelSize ?? 25);
      if (!Number.isFinite(k) || k < 1) {
        throw new Error(`seriesDecomp: kernelSize must be a positive integer, got ${params.kernelSize}`);
      }
      if (k % 2 === 0) {
        throw new Error(`seriesDecomp: kernelSize ${k} is even; the moving-average padding needs an odd kernel to keep sequence length`);
      }
      return inputShape;
    }
  },
  setAbstraction: {
    type: "setAbstraction",
    name: "PointNet++ Set Abstraction",
    icon: "\u{1F7E2}",
    category: "cv",
    defaultParams: { numPoints: 512, radius: 0.2, numSamples: 32, mlp: [64, 64, 128] },
    computeOutputShape: (inputShape, params) => {
      const numPoints = params.numPoints ?? 512;
      const mlp = Array.isArray(params.mlp) ? params.mlp : [64, 64, 128];
      const lastDim = mlp.length > 0 ? mlp[mlp.length - 1] : 128;
      if (inputShape.length < 2) return [numPoints, 128];
      return [numPoints, lastDim];
    }
  },
  sparseConv3d: {
    type: "sparseConv3d",
    name: "Submanifold Sparse Conv3D",
    icon: "\u{1F537}",
    category: "cv",
    defaultParams: { inChannels: 32, outChannels: 64, kernelSize: 3 },
    computeOutputShape: (inputShape, params) => {
      const outChannels = params.outChannels ?? 64;
      if (inputShape.length >= 4) {
        const [, d, h, w] = inputShape;
        return [outChannels, d, h, w];
      }
      return inputShape;
    }
  },
  nerfPositionalEncoding: {
    type: "nerfPositionalEncoding",
    name: "Fourier Feature Encoding",
    icon: "\u{1F308}",
    category: "utility",
    defaultParams: { numFrequencies: 10, includeInput: true },
    computeOutputShape: (inputShape, params) => {
      const numFrequencies = params.numFrequencies ?? 10;
      const includeInput = params.includeInput ?? true;
      if (inputShape.length === 0) return inputShape;
      const out = [...inputShape];
      const last = out[out.length - 1];
      out[out.length - 1] = last * (2 * numFrequencies + (includeInput ? 1 : 0));
      return out;
    }
  },
  dividedSpaceTimeAttention: {
    type: "dividedSpaceTimeAttention",
    name: "Divided Space-Time Attention",
    icon: "\u{1F39E}\uFE0F",
    category: "cv",
    defaultParams: { embedDim: 768, numHeads: 12 },
    computeOutputShape: (inputShape) => inputShape
  },
  tubeletEmbed: {
    type: "tubeletEmbed",
    name: "Tubelet Embedding (3D Patch)",
    icon: "\u{1F4F9}",
    category: "cv",
    defaultParams: { tubeletSize: [2, 16, 16], embedDim: 768 },
    computeOutputShape: (inputShape, params) => {
      const embedDim = params.embedDim ?? 768;
      const tubelet = Array.isArray(params.tubeletSize) ? params.tubeletSize : [2, 16, 16];
      if (inputShape.length >= 4) {
        const [, t, h, w] = inputShape;
        const [tt, th, tw] = tubelet;
        const numTubelets = Math.max(1, Math.floor(t / (tt || 1))) * Math.max(1, Math.floor(h / (th || 1))) * Math.max(1, Math.floor(w / (tw || 1)));
        return [numTubelets, embedDim];
      }
      return [196, embedDim];
    }
  }
};
function createComponent(type, position, id) {
  const def = componentRegistry[type];
  return {
    id: id || `comp-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    type,
    name: def.name,
    position,
    params: { ...def.defaultParams },
    inputs: [],
    outputs: []
  };
}

// src/utils/pythonStmts.ts
function parsePyStmts(source) {
  const logical = toLogicalLines(source);
  return buildTree(logical, 0, logical.length, 0);
}
function toLogicalLines(src) {
  const lines = src.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    if (raw.trim() === "" || raw.trimStart().startsWith("#")) {
      out.push({ text: raw, indent: indentWidth(raw), startLine: i, endLine: i });
      i++;
      continue;
    }
    const startLine = i;
    let combined = raw;
    let depth = bracketDelta(raw);
    let endsWithBackslash = stripComment(raw).trimEnd().endsWith("\\");
    while ((depth > 0 || endsWithBackslash) && i + 1 < lines.length) {
      i++;
      const next = lines[i];
      combined += "\n" + next;
      depth += bracketDelta(next);
      endsWithBackslash = stripComment(next).trimEnd().endsWith("\\");
    }
    out.push({
      text: combined,
      indent: indentWidth(raw),
      startLine,
      endLine: i
    });
    i++;
  }
  return out;
}
function stripComment(line) {
  let i = 0;
  let inStr = false;
  while (i < line.length) {
    const ch = line[i];
    if (inStr) {
      if (typeof inStr === "string" && inStr.length === 3) {
        if (line.slice(i, i + 3) === inStr) {
          inStr = false;
          i += 3;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      } else {
        if (ch === inStr) {
          inStr = false;
          i++;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      }
      i++;
      continue;
    }
    if (ch === "#") return line.slice(0, i);
    if (line.slice(i, i + 3) === "'''" || line.slice(i, i + 3) === '"""') {
      inStr = line.slice(i, i + 3);
      i += 3;
      continue;
    }
    if (ch === "'" || ch === '"') {
      inStr = ch;
      i++;
      continue;
    }
    i++;
  }
  return line;
}
function indentWidth(line) {
  let w = 0;
  for (const ch of line) {
    if (ch === " ") w++;
    else if (ch === "	") w += 8;
    else break;
  }
  return w;
}
function bracketDelta(line) {
  let d = 0;
  let i = 0;
  let inStr = false;
  while (i < line.length) {
    const ch = line[i];
    if (inStr) {
      if (typeof inStr === "string" && inStr.length === 3) {
        if (line.slice(i, i + 3) === inStr) {
          inStr = false;
          i += 3;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      } else {
        if (ch === inStr) {
          inStr = false;
          i++;
          continue;
        }
        if (ch === "\\") {
          i += 2;
          continue;
        }
      }
      i++;
      continue;
    }
    if (ch === "#") break;
    if (line.slice(i, i + 3) === "'''" || line.slice(i, i + 3) === '"""') {
      inStr = line.slice(i, i + 3);
      i += 3;
      continue;
    }
    if (ch === "'" || ch === '"') {
      inStr = ch;
      i++;
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") d++;
    else if (ch === ")" || ch === "]" || ch === "}") d--;
    i++;
  }
  return d;
}
function buildTree(lines, start, end, baseIndent) {
  const out = [];
  let pendingDecorators = [];
  let i = start;
  while (i < end) {
    const ln = lines[i];
    const trimmed = ln.text.trim();
    if (trimmed === "" || trimmed.startsWith("#")) {
      i++;
      continue;
    }
    if (ln.indent < baseIndent) break;
    if (ln.indent > baseIndent) {
      i++;
      continue;
    }
    if (trimmed.startsWith("@")) {
      pendingDecorators.push(trimmed.replace(/^@\s*/, ""));
      i++;
      continue;
    }
    const head = headWord(trimmed);
    const stmt = {
      kind: classifyStmt(trimmed, head),
      text: ln.text,
      indent: ln.indent,
      startLine: ln.startLine,
      endLine: ln.endLine,
      body: []
    };
    if (pendingDecorators.length > 0 && (stmt.kind === "class" || stmt.kind === "def")) {
      stmt.decorators = pendingDecorators;
    }
    pendingDecorators = [];
    if (stmt.kind === "class") {
      const m = trimmed.match(/^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:/);
      if (m) {
        stmt.name = m[1];
        stmt.bases = (m[2] || "").split(",").map((s) => s.trim()).filter(Boolean);
      }
    } else if (stmt.kind === "def") {
      const m = trimmed.match(/^(?:async\s+)?def\s+(\w+)\s*\(/);
      if (m) stmt.name = m[1];
    }
    if (isBlockOpener(stmt.kind) && trimmed.endsWith(":")) {
      const bodyStart = i + 1;
      let bodyEnd = bodyStart;
      let bodyIndent = -1;
      for (let j = bodyStart; j < end; j++) {
        const t = lines[j].text.trim();
        if (t === "" || t.startsWith("#")) continue;
        if (lines[j].indent <= ln.indent) break;
        if (bodyIndent === -1) bodyIndent = lines[j].indent;
        bodyEnd = j + 1;
      }
      if (bodyIndent !== -1) {
        stmt.body = buildTree(lines, bodyStart, bodyEnd, bodyIndent);
        stmt.endLine = lines[bodyEnd - 1].endLine;
      }
      i = bodyEnd;
    } else {
      i++;
    }
    out.push(stmt);
  }
  return out;
}
function headWord(text) {
  const m = text.match(/^[A-Za-z_]\w*/);
  return m ? m[0] : "";
}
function classifyStmt(text, head) {
  if (head === "class") return "class";
  if (head === "def" || head === "async" && /^async\s+def\b/.test(text)) return "def";
  if (head === "if" || head === "elif" || head === "else") return "if";
  if (head === "for") return "for";
  if (head === "while") return "while";
  if (head === "with") return "with";
  if (head === "try" || head === "except" || head === "finally") return "try";
  if (head === "return") return "return";
  if (head === "import" || head === "from") return "import";
  if (hasTopLevelAssign(text)) return "assign";
  return "expr";
}
function isBlockOpener(k) {
  return k === "class" || k === "def" || k === "if" || k === "for" || k === "while" || k === "with" || k === "try";
}
function hasTopLevelAssign(text) {
  let depth = 0;
  let inStr = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === inStr) inStr = false;
      else if (ch === "\\") i++;
      continue;
    }
    if (ch === "'" || ch === '"') {
      inStr = ch;
      continue;
    }
    if (ch === "#") break;
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth--;
    else if (ch === "=" && depth === 0) {
      const prev = text[i - 1];
      const next = text[i + 1];
      if (next === "=") {
        i++;
        continue;
      }
      if ("=<>!:+-*/%&|^~".includes(prev)) continue;
      return true;
    }
  }
  return false;
}
function* walk(stmts) {
  for (const s of stmts) {
    yield s;
    if (s.body.length > 0) yield* walk(s.body);
  }
}
function findMainModelClass(stmts) {
  let candidate = null;
  for (const s of walk(stmts)) {
    if (s.kind !== "class") continue;
    const hasInit = s.body.some((c) => c.kind === "def" && c.name === "__init__");
    const hasForward = s.body.some((c) => c.kind === "def" && c.name === "forward");
    if (hasInit && hasForward) candidate = s;
  }
  return candidate;
}

// src/utils/codeParser.ts
var CUSTOM_CLASS_MAP = {
  // Positional encoding
  positionalencoding: "positionalEncoding",
  posencoding: "positionalEncoding",
  posemb: "positionalEncoding",
  positionalembedding: "positionalEncoding",
  rotaryembedding: "rope",
  ropeembedding: "rope",
  // Attention — standard
  multiheadattention: "multiHeadAttention",
  multiheadattn: "multiHeadAttention",
  causalselfattention: "causalAttention",
  selfattn: "selfAttention",
  selfattention: "selfAttention",
  attention: "attention",
  crossattention: "crossModalAttention",
  // Attention — GQA / modern LLMs
  llamaattention: "groupedQueryAttention",
  mistralattention: "groupedQueryAttention",
  mixtralattention: "groupedQueryAttention",
  qwenattention: "groupedQueryAttention",
  gemmaattention: "groupedQueryAttention",
  phi3attention: "groupedQueryAttention",
  falcon7battention: "groupedQueryAttention",
  groupedqueryattention: "groupedQueryAttention",
  gqaattention: "groupedQueryAttention",
  // Feed-forward / SwiGLU MLP
  feedforward: "feedForward",
  feedforwardnetwork: "feedForward",
  ffn: "feedForward",
  mlp: "feedForward",
  llamamlp: "swiglu",
  mistralmpl: "swiglu",
  mixtralmlp: "swiglu",
  qwenmlp: "swiglu",
  gemmamlp: "swiglu",
  phi3mlp: "swiglu",
  swiglu: "swiglu",
  gatedmlp: "swiglu",
  // MoE
  mixtralsparsemoeblock: "moeLayer",
  moelayer: "moeLayer",
  expertlayer: "moeLayer",
  sparsemlp: "moeLayer",
  // Transformer block — generic
  block: "transformerBlock",
  gptblock: "transformerBlock",
  bertlayer: "transformerBlock",
  bertblock: "transformerBlock",
  encoderlayer: "transformerBlock",
  decoderlayer: "transformerBlock",
  transformerblock: "transformerBlock",
  transformerlayer: "transformerBlock",
  transformer: "transformerBlock",
  visionblock: "transformerBlock",
  // Transformer block — named LLM variants
  llamadecoderlayer: "transformerBlock",
  llamadecoderblock: "transformerBlock",
  mistraldecoderlayer: "transformerBlock",
  mixtraldecoderlayer: "transformerBlock",
  qwendecoderlayer: "transformerBlock",
  gemmadecoderlayer: "transformerBlock",
  phi3decoderlayer: "transformerBlock",
  // ViT / Vision
  visiontransformer: "transformerBlock",
  vitblock: "transformerBlock",
  vitlayer: "transformerBlock",
  patchembed: "patchEmbed",
  patchembedding: "patchEmbed",
  patchprojection: "patchEmbed",
  // SE / Squeeze-Excite
  seblock: "seBlock",
  squeezeexcitation: "seBlock",
  channelattention: "seBlock",
  // Residual
  resblock: "residual",
  residualblock: "residual",
  resnetblock: "residual",
  bottleneck: "residual"
};
function mapCustomClassToComponent(className) {
  return CUSTOM_CLASS_MAP[className.toLowerCase()] ?? null;
}
function findMainClassRange(tree) {
  const main = findMainModelClass(tree);
  if (!main) return null;
  const initStmt = main.body.find((c) => c.kind === "def" && c.name === "__init__");
  const forwardStmt = main.body.find((c) => c.kind === "def" && c.name === "forward");
  if (!initStmt || !forwardStmt) return null;
  return {
    name: main.name ?? "",
    initLine: initStmt.startLine,
    forwardLine: forwardStmt.startLine,
    endLine: main.endLine
  };
}
var MAX_EXPAND_DEPTH = 3;
function buildClassRegistry(tree) {
  const registry = /* @__PURE__ */ new Map();
  for (const s of walk(tree)) {
    if (s.kind !== "class" || !s.name) continue;
    const initStmt = s.body.find((c) => c.kind === "def" && c.name === "__init__");
    if (!initStmt) continue;
    const forwardStmt = s.body.find((c) => c.kind === "def" && c.name === "forward") ?? null;
    const methods = /* @__PURE__ */ new Map();
    for (const c of s.body) {
      if (c.kind === "def" && c.name && c.name !== "__init__" && c.name !== "forward") {
        methods.set(c.name, c);
      }
    }
    registry.set(s.name, { name: s.name, initStmt, forwardStmt, methods });
  }
  return registry;
}
function canExpandClass(className, ctx) {
  return ctx.registry.has(className) && ctx.depth < MAX_EXPAND_DEPTH && !ctx.expanding.has(className);
}
function splitTopLevelArgs(s) {
  const parts = [];
  let depth = 0;
  let cur = "";
  for (const ch of s) {
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth--;
    if (ch === "," && depth === 0) {
      parts.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}
function extractBalancedArgs(text, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];
    if (ch === "(") depth++;
    else if (ch === ")") {
      depth--;
      if (depth === 0) return text.slice(openIdx + 1, i);
    }
  }
  return null;
}
function resolveNumExpr(expr, env) {
  const t = expr.trim();
  if (/^\d+$/.test(t)) return parseInt(t, 10);
  if (/^\d*\.\d+$/.test(t)) return parseFloat(t);
  if (/^\w+$/.test(t)) return typeof env[t] === "number" ? env[t] : null;
  const m = t.match(/^(\w+)\s*([+\-*])\s*(\w+)$/);
  if (m) {
    const a = resolveNumExpr(m[1], env);
    const b = resolveNumExpr(m[3], env);
    if (a === null || b === null) return null;
    return m[2] === "+" ? a + b : m[2] === "-" ? a - b : a * b;
  }
  return null;
}
function bindCallArgs(defText, callArgsStr, callerEnv) {
  const env = {};
  const open = defText.indexOf("(");
  const sig = open >= 0 ? extractBalancedArgs(defText, open) : null;
  const params = [];
  if (sig) {
    for (const part of splitTopLevelArgs(sig)) {
      const p = part.trim();
      if (!p || p === "self" || p.startsWith("*")) continue;
      const pm = p.match(/^(\w+)\s*(?::[^=]*)?(?:=\s*(.+))?$/);
      if (!pm) continue;
      params.push(pm[1]);
      if (pm[2] !== void 0) {
        const v = resolveNumExpr(pm[2], callerEnv);
        if (v !== null) env[pm[1]] = v;
      }
    }
  }
  let pos = 0;
  for (const part of splitTopLevelArgs(callArgsStr)) {
    const a = part.trim();
    if (!a) continue;
    const kw = a.match(/^(\w+)\s*=\s*(.+)$/);
    if (kw) {
      const v = resolveNumExpr(kw[2], callerEnv);
      if (v !== null) env[kw[1]] = v;
    } else {
      const v = resolveNumExpr(a, callerEnv);
      if (v !== null && params[pos] !== void 0) env[params[pos]] = v;
      pos++;
    }
  }
  return env;
}
function resolveParamsWithEnv(params, env) {
  const out = {};
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === "string") {
      const r = resolveNumExpr(v, env);
      out[k] = r !== null ? r : v;
    } else {
      out[k] = v;
    }
  }
  return out;
}
function expandClassInstance(className, callArgsStr, ctx, callerEnv) {
  const info = ctx.registry.get(className);
  if (!info || ctx.depth >= MAX_EXPAND_DEPTH || ctx.expanding.has(className)) return [];
  const env = bindCallArgs(info.initStmt.text, callArgsStr, callerEnv);
  const innerCtx = {
    ...ctx,
    methods: info.methods,
    env,
    depth: ctx.depth + 1,
    expanding: /* @__PURE__ */ new Set([...ctx.expanding, className])
  };
  const initLayers = parseInitLayers(
    ctx.lines,
    info.initStmt.startLine,
    info.initStmt.endLine,
    innerCtx
  );
  const forwardCalls = info.forwardStmt ? parseForwardCalls(ctx.lines, info.forwardStmt.startLine, info.forwardStmt.endLine) : [];
  return orderLayersByForward(initLayers, forwardCalls).map((l) => ({
    name: l.pyName ?? l.type,
    type: l.type,
    params: resolveParamsWithEnv(l.params, env)
  }));
}
var MODULELIST_COMP_RE = /\[\s*(\w+)\s*\(([^)]*)\)\s+for\s+\w+\s+in\s+range\s*\(([^)]+)\)/;
function expandModuleListComp(compMatch, listName, initSigLine, ctx) {
  const [, className, argsStr, nExpr] = compMatch;
  let n = resolveNumExpr(nExpr, ctx?.env ?? {});
  if (n === null) {
    const defaultMatch = initSigLine.match(new RegExp(`\\b${nExpr.trim()}=(\\d+)`));
    n = defaultMatch ? parseInt(defaultMatch[1], 10) : 6;
  }
  n = Math.max(0, Math.floor(n));
  const out = [];
  const mapped = mapCustomClassToComponent(className);
  if (mapped) {
    for (let j = 0; j < n; j++) {
      out.push({ name: `${listName}_${j}`, type: mapped, params: {} });
    }
    return out;
  }
  if (ctx && canExpandClass(className, ctx)) {
    const inner = expandClassInstance(className, argsStr, ctx, ctx.env);
    for (let j = 0; j < n; j++) {
      for (const il of inner) {
        out.push({ name: `${listName}_${j}.${il.name}`, type: il.type, params: il.params });
      }
    }
  }
  return out;
}
function scanInstantiations(text, registry) {
  const out = [];
  const re = /\b(?:nn\.(\w+)|(?<![\w.])([A-Z]\w*))\s*\(/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const openIdx = m.index + m[0].length - 1;
    const args = extractBalancedArgs(text, openIdx);
    if (args === null) continue;
    if (m[1]) {
      if (m[1] !== "Sequential" && m[1] !== "ModuleList") {
        out.push({ kind: "nn", name: m[1], args });
      }
    } else if (m[2] && (registry.has(m[2]) || mapCustomClassToComponent(m[2]))) {
      out.push({ kind: "class", name: m[2], args });
    }
  }
  return out;
}
function expandHelperMethod(methodStmt, callArgsStr, attrName, ctx) {
  const env = bindCallArgs(methodStmt.text, callArgsStr, ctx.env);
  const collect = (stmts) => {
    const insts = [];
    for (const s of stmts) {
      if (s.kind === "for" && s.body.length > 0) {
        const rangeMatch = s.text.match(/\bin\s+range\s*\(([^)]+)\)/);
        const count = rangeMatch ? Math.max(0, Math.floor(resolveNumExpr(rangeMatch[1], env) ?? 1)) : 1;
        const inner = collect(s.body);
        for (let r = 0; r < count; r++) insts.push(...inner);
      } else if (s.body.length > 0) {
        insts.push(...collect(s.body));
      } else {
        insts.push(...scanInstantiations(s.text, ctx.registry));
      }
    }
    return insts;
  };
  const out = [];
  let k = 0;
  for (const inst of collect(methodStmt.body)) {
    if (inst.kind === "nn") {
      const parsed = parseLayerDefinition(`nn.${inst.name}(${inst.args})`);
      if (parsed?.type) {
        out.push({
          name: `${attrName}_${k}`,
          type: parsed.type,
          params: resolveParamsWithEnv(parsed.params, env)
        });
        k++;
      }
      continue;
    }
    const mapped = mapCustomClassToComponent(inst.name);
    if (mapped) {
      out.push({ name: `${attrName}_${k}`, type: mapped, params: {} });
      k++;
    } else if (canExpandClass(inst.name, ctx)) {
      const inner = expandClassInstance(inst.name, inst.args, ctx, env);
      if (inner.length > 0) {
        for (const il of inner) {
          out.push({ name: `${attrName}_${k}.${il.name}`, type: il.type, params: il.params });
        }
        k++;
      }
    }
  }
  return out;
}
function orderLayersByForward(initLayers, forwardCalls) {
  const allLayers = [];
  const usedLayerNames = /* @__PURE__ */ new Set();
  for (const call of forwardCalls) {
    if (call.isFunctional) {
      const componentType = mapFunctionalToComponent(call.layerName) ?? mapEinopsToComponent(call.layerName);
      if (componentType) {
        allLayers.push({ type: componentType, params: {}, isFunctional: true, pyName: call.layerName });
      }
    } else {
      const matchingLayers = initLayers.filter(
        (l) => l.name === call.layerName || l.name.startsWith(`${call.layerName}_`)
        // Sequential
      );
      if (matchingLayers.length > 1) {
        for (const layer of matchingLayers) {
          if (layer.type && !usedLayerNames.has(layer.name)) {
            allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
            usedLayerNames.add(layer.name);
          }
        }
      } else if (matchingLayers.length === 1) {
        const layer = matchingLayers[0];
        if (layer.type) {
          allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
        }
      } else {
        const layer = initLayers.find((l) => l.name === call.layerName);
        if (layer && layer.type) {
          allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
        }
      }
    }
  }
  if (allLayers.length === 0) {
    for (const layer of initLayers) {
      if (layer.type) {
        allLayers.push({ type: layer.type, params: layer.params, isFunctional: false, pyName: layer.name });
      }
    }
  }
  return allLayers;
}
function parsePyTorchCode(code) {
  try {
    const lines = code.split("\n").map((l) => l.trim());
    const tree = parsePyStmts(code);
    const mainClass = findMainClassRange(tree);
    const initStart = mainClass?.initLine ?? 0;
    const forwardStart = mainClass?.forwardLine ?? 0;
    const classEnd = mainClass?.endLine ?? lines.length - 1;
    const registry = buildClassRegistry(tree);
    const mainInfo = mainClass ? registry.get(mainClass.name) : void 0;
    const ctx = {
      lines,
      registry,
      methods: mainInfo?.methods ?? /* @__PURE__ */ new Map(),
      env: mainInfo ? bindCallArgs(mainInfo.initStmt.text, "", {}) : {},
      depth: 0,
      expanding: new Set(mainClass ? [mainClass.name] : [])
    };
    const initLayers = parseInitLayers(lines, initStart, forwardStart - 1, ctx);
    if (initLayers.length === 0) {
      return null;
    }
    const forwardCalls = parseForwardCalls(lines, forwardStart, classEnd);
    const inputShape = parseInputShape(lines);
    const components = [];
    const connections = [];
    const inputComponent = createComponent("input", { x: 200, y: 50 }, void 0);
    if (inputShape) {
      inputComponent.params.shape = inputShape;
    }
    components.push(inputComponent);
    let prevComponentId = inputComponent.id;
    const allLayers = orderLayersByForward(initLayers, forwardCalls);
    const COMPONENT_SPACING_Y = 200;
    const START_X = 200;
    const START_Y = 100;
    const calculateBestPortsForVertical = () => {
      return { fromPort: "bottom", toPort: "top" };
    };
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      if (!layer.type) continue;
      const component = createComponent(layer.type, {
        x: START_X,
        y: START_Y + i * COMPONENT_SPACING_Y
      }, void 0);
      component.params = { ...layer.params };
      if (layer.pyName) {
        const dot = layer.pyName.lastIndexOf(".");
        if (dot > 0) {
          component.scope = layer.pyName.slice(0, dot);
          component.name = layer.pyName.slice(dot + 1);
        } else {
          component.name = layer.pyName;
        }
      }
      components.push(component);
      const bestPorts = calculateBestPortsForVertical();
      const connection = {
        id: `conn-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
        from: prevComponentId,
        to: component.id,
        fromPort: bestPorts.fromPort,
        toPort: bestPorts.toPort
      };
      connections.push(connection);
      prevComponentId = component.id;
    }
    if (prevComponentId !== inputComponent.id && allLayers.length > 0) {
      const outputY = START_Y + allLayers.length * COMPONENT_SPACING_Y;
      const outputComponent = createComponent("output", { x: 200, y: outputY }, void 0);
      components.push(outputComponent);
      const lastConnection = {
        id: `conn-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
        from: prevComponentId,
        to: outputComponent.id,
        fromPort: "bottom",
        toPort: "top"
      };
      connections.push(lastConnection);
    }
    rebuildNodeIO(components, connections);
    return {
      components,
      connections,
      inputShape
    };
  } catch (error) {
    console.error("Error parsing code:", error);
    return null;
  }
}
function parseInitLayers(lines, startLine, endLine, ctx) {
  const layers = [];
  let order = 0;
  let sequentialContext = null;
  let sequentialContent = [];
  let moduleListName = null;
  let moduleListBuffer = [];
  let moduleListDepth = 0;
  for (let i = startLine; i <= endLine; i++) {
    const line = lines[i];
    if (!line.trim() || line.trim().startsWith("#")) continue;
    if (line.match(/def\s+__init__/)) continue;
    if (line.match(/^\s*def\s+\w+/) && !line.match(/def\s+__init__/)) break;
    if (moduleListName !== null) {
      moduleListBuffer.push(line);
      for (const ch of line) {
        if (ch === "[") moduleListDepth++;
        else if (ch === "]") moduleListDepth--;
      }
      if (moduleListDepth <= 0) {
        const combined = moduleListBuffer.join(" ");
        const compMatch = combined.match(MODULELIST_COMP_RE);
        if (compMatch) {
          for (const il of expandModuleListComp(compMatch, moduleListName, lines[startLine], ctx)) {
            layers.push({ name: il.name, type: il.type, params: il.params, order: order++ });
          }
        }
        moduleListName = null;
        moduleListBuffer = [];
        moduleListDepth = 0;
      }
      continue;
    }
    const moduleListMatch = line.match(/self\.(\w+)\s*=\s*nn\.ModuleList\s*\(\s*\[/);
    if (moduleListMatch) {
      const compMatch = line.match(MODULELIST_COMP_RE);
      if (compMatch) {
        for (const il of expandModuleListComp(compMatch, moduleListMatch[1], lines[startLine], ctx)) {
          layers.push({ name: il.name, type: il.type, params: il.params, order: order++ });
        }
      } else {
        moduleListName = moduleListMatch[1];
        moduleListBuffer = [line];
        for (const ch of line) {
          if (ch === "[") moduleListDepth++;
          else if (ch === "]") moduleListDepth--;
        }
      }
      continue;
    }
    const sequentialMatch = line.match(/self\.(\w+)\s*=\s*nn\.Sequential\s*\(/);
    if (sequentialMatch) {
      if (sequentialContext) {
        const seqLayers = parseSequentialContent(sequentialContent, sequentialContext.name, order);
        layers.push(...seqLayers);
        order += seqLayers.length;
      }
      sequentialContext = { name: sequentialMatch[1] };
      sequentialContent = [];
      if (line.includes(")")) {
        const content = line.match(/nn\.Sequential\s*\(([^)]+)\)/)?.[1];
        if (content) {
          sequentialContent.push(content);
          const seqLayers = parseSequentialContent(sequentialContent, sequentialContext.name, order);
          layers.push(...seqLayers);
          order += seqLayers.length;
          sequentialContext = null;
          sequentialContent = [];
        }
      }
      continue;
    }
    if (sequentialContext) {
      sequentialContent.push(line);
      if (line.includes(")")) {
        const seqLayers = parseSequentialContent(sequentialContent, sequentialContext.name, order);
        layers.push(...seqLayers);
        order += seqLayers.length;
        sequentialContext = null;
        sequentialContent = [];
      }
      continue;
    }
    if (line.startsWith("self.")) {
      const match = line.match(/self\.(\w+)\s*=\s*(.+)/);
      if (match) {
        const [, name, layerDef] = match;
        const helperMatch = layerDef.match(/^self\.(\w+)\s*\(/);
        if (helperMatch) {
          const methodStmt = ctx?.methods.get(helperMatch[1]);
          if (ctx && methodStmt) {
            const args = extractBalancedArgs(layerDef, layerDef.indexOf("(")) ?? "";
            for (const il of expandHelperMethod(methodStmt, args, name, ctx)) {
              layers.push({ name: il.name, type: il.type, params: il.params, order: order++ });
            }
          }
          continue;
        }
        const parsed = parseLayerDefinition(layerDef.trim());
        if (parsed) {
          layers.push({ name, type: parsed.type, params: parsed.params, order: order++ });
        } else {
          const customMatch = layerDef.match(/^(\w+)\s*\(/);
          if (customMatch) {
            const componentType = mapCustomClassToComponent(customMatch[1]);
            if (componentType) {
              layers.push({ name, type: componentType, params: {}, order: order++ });
            } else if (ctx && canExpandClass(customMatch[1], ctx)) {
              const args = extractBalancedArgs(layerDef, layerDef.indexOf("(")) ?? "";
              for (const il of expandClassInstance(customMatch[1], args, ctx, ctx.env)) {
                layers.push({ name: `${name}.${il.name}`, type: il.type, params: il.params, order: order++ });
              }
            }
          }
        }
      }
    }
  }
  return layers;
}
function parseSequentialContent(lines, sequentialName, startOrder) {
  const layers = [];
  let order = startOrder;
  const cleanedLines = lines.map((line) => {
    const commentIndex = line.indexOf("#");
    if (commentIndex >= 0) {
      const beforeComment = line.substring(0, commentIndex);
      const openParens = (beforeComment.match(/\(/g) || []).length;
      const closeParens = (beforeComment.match(/\)/g) || []).length;
      if (openParens > closeParens) {
        return line;
      } else {
        return beforeComment.trim();
      }
    }
    return line.trim();
  });
  const fullText = cleanedLines.join("\n");
  const layerPattern = /nn\.(\w+)\s*\([^)]*(?:\([^)]*\)[^)]*)*\)/g;
  let match;
  const matches = [];
  while ((match = layerPattern.exec(fullText)) !== null) {
    matches.push(match[0]);
  }
  if (matches.length === 0) {
    for (const line of cleanedLines) {
      const simpleMatch = line.match(/nn\.(\w+)\s*\(([^)]*)\)/);
      if (simpleMatch) {
        matches.push(simpleMatch[0]);
      }
    }
  }
  for (const layerDef of matches) {
    const parsed = parseLayerDefinition(layerDef);
    if (parsed && parsed.type) {
      layers.push({
        name: `${sequentialName}_${order - startOrder + 1}`,
        type: parsed.type,
        params: parsed.params,
        order: order++
      });
    }
  }
  return layers;
}
function parseLayerDefinition(layerDef) {
  const params = {};
  if (layerDef.includes("F.") || layerDef.includes("torch.")) {
    return null;
  }
  const linearMatch = layerDef.match(/nn\.Linear\s*\(\s*([^,]+)\s*,\s*([^,)]+)/);
  if (linearMatch) {
    const [, inFeatures, outFeatures] = linearMatch;
    params.inFeatures = parseNumberOrVariable(inFeatures);
    params.outFeatures = parseNumberOrVariable(outFeatures);
    return { type: "linear", params };
  }
  const convT2dMatch = layerDef.match(/nn\.ConvTranspose2d\s*\(([^)]+)\)/);
  if (convT2dMatch) {
    const args = parseArguments(convT2dMatch[1]);
    params.inChannels = parseNumberOrVariable(args.in_channels || args[0]);
    params.outChannels = parseNumberOrVariable(args[1] || args.out_channels || args[0]);
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[2] || "3");
    params.stride = parseNumberOrVariable(args.stride || args[3] || "1");
    params.padding = parseNumberOrVariable(args.padding || args[4] || "0");
    if (args.output_padding !== void 0) params.outputPadding = parseNumberOrVariable(args.output_padding);
    if (args.dilation !== void 0) params.dilation = parseNumberOrVariable(args.dilation);
    if (args.groups !== void 0) params.groups = parseNumberOrVariable(args.groups);
    return { type: "transposeConv2d", params };
  }
  const conv2dMatch = layerDef.match(/nn\.Conv2d\s*\(([^)]+)\)/);
  if (conv2dMatch) {
    const args = parseArguments(conv2dMatch[1]);
    const pk = parseNumberOrVariable(args.kernel_size || args[2] || "0");
    const ps = parseNumberOrVariable(args.stride || args[3] || "1");
    const pInC = parseNumberOrVariable(args.in_channels || args[0] || "3");
    const pOutC = parseNumberOrVariable(args.out_channels || args[1] || "768");
    if (typeof pk === "number" && pk > 4 && pk === ps && typeof pInC === "number" && pInC <= 4) {
      params.patchSize = pk;
      params.embedDim = pOutC;
      params.inChans = pInC;
      return { type: "patchEmbed", params };
    }
    params.inChannels = parseNumberOrVariable(args.in_channels || args[0]);
    params.outChannels = parseNumberOrVariable(args[1] || args.out_channels || args[0]);
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[2] || "3");
    params.stride = parseNumberOrVariable(args.stride || args[3] || "1");
    params.padding = parseNumberOrVariable(args.padding || args[4] || "0");
    if (args.dilation !== void 0) params.dilation = parseNumberOrVariable(args.dilation);
    if (args.groups !== void 0) params.groups = parseNumberOrVariable(args.groups);
    return { type: "conv2d", params };
  }
  const conv1dMatch = layerDef.match(/nn\.Conv1d\s*\(([^)]+)\)/);
  if (conv1dMatch) {
    const args = parseArguments(conv1dMatch[1]);
    params.inChannels = parseNumberOrVariable(args.in_channels || args[0]);
    params.outChannels = parseNumberOrVariable(args[1] || args.out_channels);
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[2] || "3");
    params.stride = parseNumberOrVariable(args.stride || args[3] || "1");
    params.padding = parseNumberOrVariable(args.padding || args[4] || "0");
    if (args.dilation !== void 0) params.dilation = parseNumberOrVariable(args.dilation);
    if (args.groups !== void 0) params.groups = parseNumberOrVariable(args.groups);
    return { type: "conv1d", params };
  }
  const maxpoolMatch = layerDef.match(/nn\.MaxPool2d\s*\(([^)]*)\)/);
  if (maxpoolMatch) {
    const args = parseArguments(maxpoolMatch[1] || "");
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[0] || "2");
    params.stride = parseNumberOrVariable(args.stride || args[1] || args.kernel_size || "2");
    return { type: "maxpool2d", params };
  }
  const avgpoolMatch = layerDef.match(/nn\.AvgPool2d\s*\(([^)]*)\)/);
  if (avgpoolMatch) {
    const args = parseArguments(avgpoolMatch[1] || "");
    params.kernelSize = parseNumberOrVariable(args.kernel_size || args[0] || "2");
    params.stride = parseNumberOrVariable(args.stride || args[1] || args.kernel_size || "2");
    return { type: "avgpool2d", params };
  }
  const dropoutMatch = layerDef.match(/nn\.Dropout\s*\(([^)]*)\)/);
  if (dropoutMatch) {
    const args = parseArguments(dropoutMatch[1] || "");
    params.p = parseNumberOrVariable(args.p || args[0] || "0.5");
    return { type: "dropout", params };
  }
  const batchNormMatch = layerDef.match(/nn\.BatchNorm(?:2d|3d)\s*\(([^)]*)\)/);
  if (batchNormMatch) {
    return { type: "batchNorm", params };
  }
  if (layerDef.match(/nn\.ReLU\s*\(/)) {
    return { type: "relu", params: {} };
  }
  if (layerDef.match(/nn\.Sigmoid\s*\(/)) return { type: "sigmoid", params: {} };
  if (layerDef.match(/nn\.Tanh\s*\(/)) return { type: "tanh", params: {} };
  if (layerDef.match(/nn\.GELU\s*\(/)) return { type: "gelu", params: {} };
  if (layerDef.match(/nn\.SiLU\s*\(/)) return { type: "swish", params: {} };
  if (layerDef.match(/nn\.Swish\s*\(/)) return { type: "swish", params: {} };
  if (layerDef.match(/nn\.Softmax\s*\(/)) return { type: "softmax", params: {} };
  const leakyReluMatch = layerDef.match(/nn\.LeakyReLU\s*\(([^)]*)\)/);
  if (leakyReluMatch) {
    const args = parseArguments(leakyReluMatch[1] || "");
    params.negativeSlope = parseNumberOrVariable(args.negative_slope || args[0] || "0.01");
    return { type: "leakyRelu", params };
  }
  const layerNormMatch = layerDef.match(/nn\.LayerNorm\s*\(([^)]+)\)/);
  if (layerNormMatch) {
    const args = parseArguments(layerNormMatch[1]);
    params.normalizedShape = parseNumberOrVariable(args.normalized_shape || args[0] || "768");
    return { type: "layerNorm", params };
  }
  const rmsNormMatch = layerDef.match(/nn\.RMSNorm\s*\(([^)]+)\)/);
  if (rmsNormMatch) {
    const args = parseArguments(rmsNormMatch[1]);
    params.normalizedShape = parseNumberOrVariable(args.normalized_shape || args[0] || "768");
    return { type: "rmsNorm", params };
  }
  const groupNormMatch = layerDef.match(/nn\.GroupNorm\s*\(([^)]+)\)/);
  if (groupNormMatch) {
    const args = parseArguments(groupNormMatch[1]);
    params.numGroups = parseNumberOrVariable(args.num_groups || args[0] || "32");
    params.numChannels = parseNumberOrVariable(args.num_channels || args[1]);
    return { type: "groupNorm", params };
  }
  const batchNorm1dMatch = layerDef.match(/nn\.BatchNorm1d\s*\(([^)]*)\)/);
  if (batchNorm1dMatch) {
    return { type: "batchNorm", params };
  }
  const embeddingMatch = layerDef.match(/nn\.Embedding\s*\(([^)]+)\)/);
  if (embeddingMatch) {
    const args = parseArguments(embeddingMatch[1]);
    params.vocabSize = parseNumberOrVariable(args.num_embeddings || args[0] || "10000");
    params.embeddingDim = parseNumberOrVariable(args.embedding_dim || args[1] || "128");
    return { type: "embedding", params };
  }
  const telMatch = layerDef.match(/nn\.TransformerEncoderLayer\s*\(([^)]+)\)/);
  if (telMatch) {
    const args = parseArguments(telMatch[1]);
    params.embedDim = parseNumberOrVariable(args.d_model || args[0] || "512");
    params.numHeads = parseNumberOrVariable(args.nhead || args[1] || "8");
    params.ffDim = parseNumberOrVariable(args.dim_feedforward || args[2] || "2048");
    return { type: "transformerBlock", params };
  }
  const mhaMatch = layerDef.match(/nn\.MultiheadAttention\s*\(([^)]+)\)/);
  if (mhaMatch) {
    const args = parseArguments(mhaMatch[1]);
    params.hiddenDim = parseNumberOrVariable(args.embed_dim || args[0] || "512");
    params.numHeads = parseNumberOrVariable(args.num_heads || args[1] || "8");
    return { type: "multiHeadAttention", params };
  }
  const captureRecurrentExtras = (args) => {
    const numLayers = args.num_layers ?? args[2];
    if (numLayers !== void 0) params.numLayers = parseNumberOrVariable(String(numLayers));
    if (args.bidirectional !== void 0) {
      params.bidirectional = args.bidirectional === "True" || args.bidirectional === true;
    }
  };
  const lstmMatch = layerDef.match(/nn\.LSTM\s*\(([^)]+)\)/);
  if (lstmMatch) {
    const args = parseArguments(lstmMatch[1]);
    params.inputSize = parseNumberOrVariable(args.input_size || args[0] || "128");
    params.hiddenSize = parseNumberOrVariable(args.hidden_size || args[1] || "128");
    captureRecurrentExtras(args);
    return { type: "lstm", params };
  }
  const gruMatch = layerDef.match(/nn\.GRU\s*\(([^)]+)\)/);
  if (gruMatch) {
    const args = parseArguments(gruMatch[1]);
    params.inputSize = parseNumberOrVariable(args.input_size || args[0] || "128");
    params.hiddenSize = parseNumberOrVariable(args.hidden_size || args[1] || "128");
    captureRecurrentExtras(args);
    return { type: "gru", params };
  }
  const rnnMatch = layerDef.match(/nn\.RNN\s*\(([^)]+)\)/);
  if (rnnMatch) {
    const args = parseArguments(rnnMatch[1]);
    params.inputSize = parseNumberOrVariable(args.input_size || args[0] || "128");
    params.hiddenSize = parseNumberOrVariable(args.hidden_size || args[1] || "128");
    captureRecurrentExtras(args);
    return { type: "rnn", params };
  }
  if (layerDef.match(/nn\.Flatten\s*\(/)) {
    return { type: "flatten", params: {} };
  }
  const adaptiveAvgMatch = layerDef.match(/nn\.AdaptiveAvgPool2d\s*\(([^)]*)\)/);
  if (adaptiveAvgMatch) {
    const inner = adaptiveAvgMatch[1].trim().replace(/^\(+/, "").replace(/\)+$/, "");
    const nums = inner.split(",").map((t) => Number(t.trim())).filter((n) => Number.isFinite(n));
    if (nums.length === 0 || nums.every((n) => n === 1)) {
      return { type: "globalAvgPool2d", params: {} };
    }
    params.outputSize = nums.length === 1 ? nums[0] : nums;
    return { type: "adaptiveAvgPool2d", params };
  }
  const upsampleMatch = layerDef.match(/nn\.Upsample\s*\(([^)]+)\)/);
  if (upsampleMatch) {
    const args = parseArguments(upsampleMatch[1]);
    params.scaleFactor = parseNumberOrVariable(args.scale_factor || args[0] || "2");
    return { type: "upsample", params };
  }
  return null;
}
function parseArguments(argsStr) {
  const result = {};
  const positional = [];
  if (!argsStr.trim()) return result;
  const parts = argsStr.split(",").map((p) => p.trim());
  for (const part of parts) {
    if (part.includes("=")) {
      const [key, value] = part.split("=").map((p) => p.trim());
      result[key] = parseNumberOrVariable(value);
    } else {
      positional.push(parseNumberOrVariable(part));
      result[positional.length - 1] = parseNumberOrVariable(part);
    }
  }
  return result;
}
function parseNumberOrVariable(value) {
  if (!value) return void 0;
  const num = Number(value);
  if (!isNaN(num)) {
    return num;
  }
  return value;
}
function parseForwardCalls(lines, startLine, endLine) {
  const calls = [];
  const seenForLoop = /* @__PURE__ */ new Set();
  for (let i = startLine; i <= endLine; i++) {
    const line = lines[i];
    if (!line.trim() || line.trim().startsWith("#")) continue;
    if (line.match(/def\s+forward/)) continue;
    if (line.match(/^\s*def\s+\w+/) || line.match(/^class\s+/)) break;
    const forLoopMatch = line.match(/^\s*for\s+\w+\s+in\s+self\.(\w+)/);
    if (forLoopMatch) {
      const layerName = forLoopMatch[1];
      if (!seenForLoop.has(layerName)) {
        calls.push({ layerName, isFunctional: false });
        seenForLoop.add(layerName);
      }
      continue;
    }
    const selfCalls = [...line.matchAll(/self\.(\w+)\s*\(/g)].map((m) => m[1]);
    for (const name of [...selfCalls].reverse()) {
      calls.push({ layerName: name, isFunctional: false });
    }
    for (const m of line.matchAll(/(?:F\.|torch\.)(\w+)\s*\(/g)) {
      const funcName = m[1];
      const componentType = mapFunctionalToComponent(funcName);
      if (componentType) {
        calls.push({ layerName: funcName, isFunctional: true });
      }
    }
    for (const m of line.matchAll(/(?:einops\.)?(rearrange|repeat|reduce)\s*\(/g)) {
      const funcName = m[1];
      const componentType = mapEinopsToComponent(funcName);
      if (componentType) {
        calls.push({ layerName: funcName, isFunctional: true });
      }
    }
  }
  return calls;
}
function mapFunctionalToComponent(funcName) {
  const mapping = {
    // Activations
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "gelu": "gelu",
    "silu": "swish",
    "leaky_relu": "leakyRelu",
    "elu": "elu",
    "selu": "selu",
    "prelu": "prelu",
    "mish": "mish",
    "hardswish": "hardSwish",
    "hard_swish": "hardSwish",
    "glu": "glu",
    "softmax": "softmax",
    // Regularization
    "dropout": "dropout",
    // Pooling
    "max_pool2d": "maxpool2d",
    "avg_pool2d": "avgpool2d",
    "adaptive_avg_pool2d": "globalAvgPool2d",
    // Normalization
    "layer_norm": "layerNorm",
    "rms_norm": "rmsNorm",
    "batch_norm": "batchNorm",
    // Shape ops
    "flatten": "flatten",
    "interpolate": "interpolate",
    "upsample": "upsample",
    "pad": "pad",
    // Attention (PyTorch 2.0+ canonical; can't recover head/dim from call site,
    // user fills those in on the canvas)
    "scaled_dot_product_attention": "attention"
  };
  return mapping[funcName] || null;
}
function mapEinopsToComponent(funcName) {
  const mapping = {
    "rearrange": "reshape",
    "repeat": "reshape",
    "reduce": "reshape"
  };
  return mapping[funcName] || null;
}
function parseInputShape(lines) {
  for (const line of lines) {
    const match = line.match(/Input shape:\s*(\[[\d,\s]+\])/);
    if (match) {
      try {
        return JSON.parse(match[1]);
      } catch (e) {
      }
    }
  }
  return void 0;
}

// src/utils/lintThresholds.ts
var HIGH_DROPOUT_P = 0.65;
var DEEP_NO_RESIDUAL_MIN_LAYERS = 8;
var DEEP_NO_NORM_MIN_LAYERS = 7;
var VANISHING_MIN_LAYERS = 5;
var LARGE_ACTIVATION_MAX_ELEMENTS = 5e7;
var SWIGLU_RATIO_MIN = 2;
var SWIGLU_RATIO_MAX = 5;
var LARGE_LINEAR_MAX_PARAMS = 1e9;
var KV_BUDGET_CONTEXT_TOKENS = 8192;
var KV_BUDGET_MAX_GB = 4;
var SCALED_INIT_MIN_ATTENTION_LAYERS = 8;

// src/utils/weightInitAdvisor.ts
var ADVICE = {
  // ── Linear / Dense ────────────────────────────────────────────────────────
  linear: {
    method: "Kaiming Uniform",
    formula: "U(\u2212\u221A(6/fan_in), \u221A(6/fan_in))",
    reason: "Default for layers followed by ReLU; preserves activation variance.",
    pyTorch: 'nn.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")'
  },
  // ── Convolutions ─────────────────────────────────────────────────────────
  conv2d: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "He init keeps variance stable through ReLU activations.",
    pyTorch: 'nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")'
  },
  conv1d: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "Same as Conv2d: He init for ReLU-based feature extraction.",
    pyTorch: 'nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")'
  },
  depthwiseConv2d: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "Depthwise conv has fan_in = kernel_h \xD7 kernel_w; apply He init.",
    pyTorch: 'nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")'
  },
  // ── Attention / Transformer ───────────────────────────────────────────────
  multiHeadAttention: {
    method: "Xavier Normal",
    formula: "N(0, \u221A(2 / (fan_in + fan_out)))",
    reason: "Balanced input/output variance; standard for attention projections.",
    pyTorch: "nn.init.xavier_normal_(w)"
  },
  causalAttention: {
    method: "GPT-2 style (scaled Normal)",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "Scales down residual projections by depth to prevent explosion.",
    pyTorch: "nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))"
  },
  groupedQueryAttention: {
    method: "GPT-2 style (scaled Normal)",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "Matches LLaMA/Mistral init: residual outputs scaled by depth.",
    pyTorch: "nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))"
  },
  transformerBlock: {
    method: "Scaled Normal (residual)",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "Residual path init must scale with depth to keep output stable.",
    pyTorch: "nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))"
  },
  feedForward: {
    method: "Kaiming Normal + small output",
    formula: "N(0, 0.02 / \u221A(2L)) for out proj",
    reason: "Input proj: He init; output proj scaled down as residual stream.",
    pyTorch: "nn.init.kaiming_normal_(fc1.weight); nn.init.normal_(fc2.weight, std=0.02/\u221A(2L))"
  },
  swiglu: {
    method: "Scaled Normal",
    formula: "N(0, 0.02 / \u221A(2L))",
    reason: "LLaMA-style: down_proj uses scaled init; gate/up use standard.",
    pyTorch: "nn.init.normal_(down_proj.weight, std=0.02/math.sqrt(2*n_layers))"
  },
  // ── Embeddings ───────────────────────────────────────────────────────────
  embedding: {
    method: "Normal (small)",
    formula: "N(0, 1/\u221AD)",
    reason: "Small init prevents embedding norms from dominating early training.",
    pyTorch: "nn.init.normal_(embed.weight, std=1/math.sqrt(d_model))"
  },
  // ── Normalization (biases, gains) ────────────────────────────────────────
  layerNorm: {
    method: "Scale=1, Bias=0",
    formula: "\u03B3=1, \u03B2=0",
    reason: "Identity init so early gradients flow unimpeded.",
    pyTorch: "nn.init.ones_(ln.weight); nn.init.zeros_(ln.bias)"
  },
  rmsNorm: {
    method: "Scale=1",
    formula: "\u03B3=1",
    reason: "RMSNorm has no bias; init scale to 1 for identity pass-through.",
    pyTorch: "nn.init.ones_(rms.weight)"
  },
  batchNorm: {
    method: "Scale=1, Bias=0",
    formula: "\u03B3=1, \u03B2=0",
    reason: "Same as LayerNorm: identity at init lets batch statistics stabilize.",
    pyTorch: "nn.init.ones_(bn.weight); nn.init.zeros_(bn.bias)"
  },
  groupNorm: {
    method: "Scale=1, Bias=0",
    formula: "\u03B3=1, \u03B2=0",
    reason: "Standard normalisation layer init.",
    pyTorch: "nn.init.ones_(gn.weight); nn.init.zeros_(gn.bias)"
  },
  // ── RNN ──────────────────────────────────────────────────────────────────
  lstm: {
    method: "Orthogonal (hidden) + Xavier (input)",
    formula: "W_h = orthonormal; W_x = Xavier",
    reason: "Orthogonal hidden weights preserve gradient norms over long sequences.",
    pyTorch: "nn.init.orthogonal_(lstm.weight_hh_l0); nn.init.xavier_uniform_(lstm.weight_ih_l0)"
  },
  gru: {
    method: "Orthogonal (hidden) + Xavier (input)",
    formula: "W_h = orthonormal; W_x = Xavier",
    reason: "Same as LSTM: orthogonal init prevents vanishing/exploding in GRU.",
    pyTorch: "nn.init.orthogonal_(gru.weight_hh_l0); nn.init.xavier_uniform_(gru.weight_ih_l0)"
  },
  // ── Output heads ─────────────────────────────────────────────────────────
  patchEmbed: {
    method: "Kaiming Normal",
    formula: "N(0, \u221A(2 / fan_in))",
    reason: "PatchEmbed is a Conv2d projection; He init standard.",
    pyTorch: "nn.init.kaiming_normal_(patch_embed.proj.weight)"
  },
  // ── Activations (no weights, but biases) ─────────────────────────────────
  prelu: {
    method: "PReLU slope init",
    formula: "a = 0.25",
    reason: "PyTorch default: close to leaky ReLU for stable start.",
    pyTorch: "nn.init.constant_(prelu.weight, 0.25)"
  }
};
function getInitAdvice(type) {
  return ADVICE[type] ?? null;
}

// src/utils/architectureAdvisor.ts
var largestDivisorAtMost = (n, max) => {
  for (let d = Math.min(max, n); d >= 1; d--) if (n % d === 0) return d;
  return 1;
};
var ACTIVATION_TYPES = /* @__PURE__ */ new Set([
  "relu",
  "gelu",
  "swish",
  "sigmoid",
  "tanh",
  "leakyRelu",
  "softmax",
  "silu"
]);
var NORM_TYPES = /* @__PURE__ */ new Set([
  "batchNorm",
  "layerNorm",
  "instanceNorm",
  "groupNorm",
  "rmsNorm"
]);
var DEEP_LAYER_TYPES = /* @__PURE__ */ new Set([
  "conv2d",
  "conv1d",
  "conv3d",
  "linear",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d"
]);
var RESIDUAL_TYPES = /* @__PURE__ */ new Set([
  "residual",
  "skipConnection",
  "add"
]);
var ATTENTION_TYPES = /* @__PURE__ */ new Set([
  "attention",
  "selfAttention",
  "multiHeadAttention",
  "groupedQueryAttention",
  "causalAttention",
  "mla"
]);
var PE_TYPES = /* @__PURE__ */ new Set([
  // A learned relative bias is how T5 (and every model that copied it) tells
  // attention where tokens are. Leaving it out of this set made R14 warn that
  // "attention is permutation-invariant" on models whose whole positional
  // scheme was sitting on the canvas.
  "positionalEncoding",
  "learnedPositionalEmbedding",
  "rope",
  "alibi",
  "relativePositionBias"
]);
var VANISHING_ACTIVATIONS = /* @__PURE__ */ new Set(["sigmoid", "tanh"]);
var noInputNode = (model) => {
  if (model.components.length > 0 && !model.components.some((c) => c.type === "input")) {
    return [{
      id: "no-input-node",
      ruleId: "no-input-node",
      // Warning, not error: a missing Input is the normal state while building
      // (e.g. right after dropping a block) and is trivially fixed, so it should
      // not raise a red "error" on a fresh, in-progress graph.
      severity: "warning",
      category: "structure",
      title: "No Input node",
      message: "The model has no Input node. Without an Input layer, tensor shapes cannot be propagated and generated code will be incomplete.",
      affectedIds: [],
      suggestion: "Drag an Input layer from the I/O section of the component palette."
    }];
  }
  return [];
};
var noOutputNode = (model) => {
  if (model.components.length > 1 && !model.components.some((c) => c.type === "output")) {
    return [{
      id: "no-output-node",
      ruleId: "no-output-node",
      severity: "warning",
      category: "structure",
      title: "No Output node",
      message: "The model has no Output node. The code generator won't know where the forward pass terminates.",
      affectedIds: [],
      suggestion: "Connect the last layer to an Output node."
    }];
  }
  return [];
};
var isolatedComponents = (model) => {
  if (model.components.length < 2) return [];
  const connected = /* @__PURE__ */ new Set();
  model.connections.forEach((c) => {
    connected.add(c.from);
    connected.add(c.to);
  });
  const isolated = model.components.filter((c) => !connected.has(c.id));
  if (isolated.length === 0) return [];
  const names = isolated.map((c) => `"${c.name}"`).join(", ");
  return [{
    id: "isolated-components",
    ruleId: "isolated-components",
    severity: "warning",
    category: "structure",
    title: `${isolated.length} isolated layer${isolated.length !== 1 ? "s" : ""}`,
    message: `${names} ${isolated.length === 1 ? "has" : "have"} no connections and will be excluded from generated code.`,
    affectedIds: isolated.map((c) => c.id),
    suggestion: "Connect these layers to the graph or delete them."
  }];
};
var deadEnds = (model) => {
  const hasOutgoing = new Set(model.connections.map((c) => c.from));
  const issues = [];
  for (const comp of model.components) {
    if (comp.type === "output") continue;
    const hasIn = model.connections.some((c) => c.to === comp.id);
    if (hasIn && !hasOutgoing.has(comp.id)) {
      issues.push({
        id: `dead-end-${comp.id}`,
        ruleId: "dead-end",
        severity: "warning",
        category: "structure",
        title: `Dead-end: "${comp.name}"`,
        message: `"${comp.name}" receives input but its output is not connected. This layer will be unreachable in the forward pass.`,
        affectedIds: [comp.id],
        suggestion: "Connect the output forward, or add an Output node if this is the final layer."
      });
    }
  }
  return issues;
};
var bnAfterActivation = (model) => {
  const issues = [];
  for (const conn of model.connections) {
    const from = model.components.find((c) => c.id === conn.from);
    const to = model.components.find((c) => c.id === conn.to);
    if (!from || !to) continue;
    if (ACTIVATION_TYPES.has(from.type) && NORM_TYPES.has(to.type)) {
      issues.push({
        id: `bn-after-act-${conn.id}`,
        ruleId: "bn-after-activation",
        severity: "warning",
        category: "ordering",
        title: "Normalization after activation",
        message: `"${to.name}" (${to.type}) follows "${from.name}" (activation). The standard pre-activation order is Conv/Linear \u2192 Norm \u2192 Activation. Normalizing post-activation limits expressivity.`,
        affectedIds: [from.id, to.id],
        suggestion: "Move the normalization layer before the activation function."
      });
    }
  }
  return issues;
};
var dropoutBeforeBN = (model) => {
  const issues = [];
  for (const conn of model.connections) {
    const from = model.components.find((c) => c.id === conn.from);
    const to = model.components.find((c) => c.id === conn.to);
    if (!from || !to) continue;
    if (from.type === "dropout" && NORM_TYPES.has(to.type)) {
      issues.push({
        id: `dropout-bn-${conn.id}`,
        ruleId: "dropout-before-bn",
        severity: "info",
        category: "ordering",
        title: "Dropout before normalization",
        message: `"${from.name}" \u2192 "${to.name}": BatchNorm re-normalizes the random zeros introduced by Dropout, nullifying most of its regularization effect.`,
        affectedIds: [from.id, to.id],
        suggestion: "Reorder to Conv \u2192 BN \u2192 Activation \u2192 Dropout."
      });
    }
  }
  return issues;
};
var outputActivation = (model) => {
  const outputNode = model.components.find((c) => c.type === "output");
  if (!outputNode) return [];
  const issues = [];
  for (const conn of model.connections.filter((c) => c.to === outputNode.id)) {
    const from = model.components.find((c) => c.id === conn.from);
    if (!from) continue;
    if (from.type === "softmax" || from.type === "sigmoid") {
      issues.push({
        id: `output-act-${conn.id}`,
        ruleId: "output-activation",
        severity: "info",
        category: "ordering",
        title: `Explicit ${from.type} before Output`,
        message: `"${from.name}" feeds directly into Output. PyTorch's nn.CrossEntropyLoss already applies log-softmax internally, an explicit Softmax causes double-application and degrades training stability.`,
        affectedIds: [from.id, outputNode.id],
        suggestion: "Remove Softmax/Sigmoid for training. Restore it in a separate inference wrapper or ONNX export.",
        fix: {
          kind: "delete-component",
          componentId: from.id,
          label: `Remove "${from.name}" and reconnect the graph`
        }
      });
    }
  }
  return issues;
};
var bnAtOutput = (model) => {
  const outputNode = model.components.find((c) => c.type === "output");
  if (!outputNode) return [];
  const issues = [];
  for (const conn of model.connections.filter((c) => c.to === outputNode.id)) {
    const from = model.components.find((c) => c.id === conn.from);
    if (!from) continue;
    if (NORM_TYPES.has(from.type)) {
      issues.push({
        id: `bn-output-${conn.id}`,
        ruleId: "bn-at-output",
        severity: "warning",
        category: "ordering",
        title: "Normalization immediately before Output",
        message: `"${from.name}" (${from.type}) is the last layer before Output. Normalizing the raw logits constrains the output range and breaks standard loss functions.`,
        affectedIds: [from.id, outputNode.id],
        suggestion: "Move normalization before the final Linear/Conv layer."
      });
    }
  }
  return issues;
};
var deepNoResidual = (model) => {
  const deepLayers = model.components.filter((c) => DEEP_LAYER_TYPES.has(c.type));
  if (deepLayers.length < DEEP_NO_RESIDUAL_MIN_LAYERS) return [];
  if (model.components.some((c) => RESIDUAL_TYPES.has(c.type))) return [];
  return [{
    id: "deep-no-residual",
    ruleId: "deep-no-residual",
    severity: "warning",
    category: "pattern",
    title: `${deepLayers.length}-layer network without skip connections`,
    message: `${deepLayers.length} conv/linear layers detected but no residual (Add/Skip) layers. Networks deeper than ${DEEP_NO_RESIDUAL_MIN_LAYERS} layers are highly prone to vanishing gradients without skip connections.`,
    affectedIds: [],
    suggestion: "Add Residual or Add layers every 2-4 layers (ResNet-style). For transformers, use the built-in TransformerBlock which includes residuals."
  }];
};
var attentionNoPE = (model) => {
  const attnNodes = model.components.filter((c) => ATTENTION_TYPES.has(c.type));
  if (attnNodes.length === 0) return [];
  if (model.components.some((c) => PE_TYPES.has(c.type))) return [];
  return [{
    id: "attention-no-pe",
    ruleId: "attention-no-pe",
    severity: "warning",
    category: "pattern",
    title: "Attention without positional encoding",
    message: `${attnNodes.length} attention layer(s) present but no positional encoding found. Attention is permutation-invariant, without position information the model cannot distinguish token order.`,
    affectedIds: attnNodes.map((c) => c.id),
    suggestion: "Add a PositionalEncoding (sinusoidal) or RoPE layer before the first attention layer."
  }];
};
var vanishingGradientRisk = (model) => {
  const nonIO = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  if (nonIO.length < VANISHING_MIN_LAYERS) return [];
  return model.components.filter((c) => VANISHING_ACTIVATIONS.has(c.type)).map((c) => ({
    id: `vanishing-${c.id}`,
    ruleId: "vanishing-gradient",
    severity: "info",
    category: "pattern",
    title: `Vanishing gradient risk: "${c.name}"`,
    message: `${c.type === "sigmoid" ? "Sigmoid" : "Tanh"} saturates to [0,1] / [-1,1], and its gradient approaches zero for large inputs. In networks deeper than ${VANISHING_MIN_LAYERS} layers, this halts learning in early layers.`,
    affectedIds: [c.id],
    suggestion: "Use ReLU, GELU, or SiLU for hidden layers. Keep Sigmoid only at binary classification outputs; Tanh in specific contexts (GAN generators, LSTM gates)."
  }));
};
var longestUnnormalizedConvRun = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const next = /* @__PURE__ */ new Map();
  for (const cn of model.connections) next.set(cn.from, [...next.get(cn.from) ?? [], cn.to]);
  const input = model.components.find((c) => c.type === "input");
  const seen = /* @__PURE__ */ new Set();
  let frontier = input ? [input.id] : [];
  let run = [];
  let best = [];
  while (frontier.length > 0) {
    const id = frontier.shift();
    if (seen.has(id)) continue;
    seen.add(id);
    const c = byId.get(id);
    if (c) {
      if (c.type === "conv2d" || c.type === "conv1d" || c.type === "conv3d") {
        run = [...run, c.id];
        if (run.length > best.length) best = run;
      } else if (NORM_TYPES.has(c.type)) {
        run = [];
      }
    }
    frontier = [...frontier, ...next.get(id) ?? []];
  }
  return { run: best.length, ids: best };
};
var DEEP_CONV_RUN_WARN = 5;
var deepNoNorm = (model) => {
  const issues = [];
  const { run, ids } = longestUnnormalizedConvRun(model);
  if (run >= DEEP_CONV_RUN_WARN) {
    issues.push({
      id: "deep-no-norm-conv-run",
      ruleId: "deep-no-norm",
      severity: "warning",
      category: "pattern",
      title: `${run} consecutive conv layers with no normalization`,
      message: `A conv stack this deep with no normalization between the layers trains poorly on realistic budgets. Measured in our production training corpus: designs with a run of ${run >= 6 ? run : 6} reached 15-27% of the reference accuracy on the same budget; shallower or normalized stacks reached 93-100%.`,
      affectedIds: ids,
      suggestion: "Insert BatchNorm every 2-3 conv layers (or GroupNorm for small batches), and downsample with pooling or strides as depth grows."
    });
  }
  const nonIO = model.components.filter((c) => c.type !== "input" && c.type !== "output");
  if (nonIO.length >= DEEP_NO_NORM_MIN_LAYERS && !model.components.some((c) => NORM_TYPES.has(c.type)) && run < DEEP_CONV_RUN_WARN) {
    issues.push({
      id: "deep-no-norm",
      ruleId: "deep-no-norm",
      severity: "info",
      category: "pattern",
      title: "No normalization in deep network",
      message: `${nonIO.length} layers with no BatchNorm, LayerNorm, or GroupNorm. Without normalization, activations can explode or vanish across layers, causing slow or unstable training.`,
      affectedIds: [],
      suggestion: "Add BatchNorm after Conv2d (CV tasks), LayerNorm after attention/FFN (NLP/LLM), or GroupNorm for small batch sizes."
    });
  }
  return issues;
};
var highDropout = (model) => {
  return model.components.filter((c) => c.type === "dropout" && (c.params.p ?? 0.5) > HIGH_DROPOUT_P).map((c) => ({
    id: `high-dropout-${c.id}`,
    ruleId: "high-dropout",
    severity: "warning",
    category: "performance",
    title: `Excessive dropout: p=${c.params.p}`,
    message: `"${c.name}" drops ${Math.round((c.params.p ?? 0.5) * 100)}% of activations per forward pass. Rates above ${HIGH_DROPOUT_P} introduce so much noise that the model cannot learn stable representations.`,
    affectedIds: [c.id],
    suggestion: "Use p \u2208 [0.1, 0.5] for most hidden layers. p=0.1-0.2 for conv layers; p=0.3-0.5 for fully-connected layers."
  }));
};
var largeActivation = (model) => {
  const offenders = [];
  for (const comp of model.components) {
    if (!comp.inputShape || !Array.isArray(comp.inputShape)) continue;
    const def = componentRegistry[comp.type];
    if (!def) continue;
    try {
      const outShape = def.computeOutputShape(comp.inputShape, comp.params);
      if (!Array.isArray(outShape)) continue;
      const elements = outShape.reduce((a, b) => a * b, 1);
      if (elements > LARGE_ACTIVATION_MAX_ELEMENTS) offenders.push({ comp, elements });
    } catch {
    }
  }
  if (offenders.length === 0) return [];
  offenders.sort((a, b) => b.elements - a.elements);
  const top = offenders[0];
  const mb = Math.round(top.elements * 4 / 1048576);
  const others = offenders.length - 1;
  return [{
    id: `large-act-${top.comp.id}`,
    ruleId: "large-activation",
    severity: "warning",
    category: "performance",
    title: others > 0 ? `Large activations: ~${mb} MB/sample across ${offenders.length} layers` : `Large activation: ~${mb} MB/sample`,
    message: others > 0 ? `"${top.comp.name}" outputs ${(top.elements / 1e6).toFixed(1)}M elements (~${mb} MB float32/sample), and ${others} more layer${others !== 1 ? "s" : ""} carry similarly large tensors downstream of the same oversized dimension. At batch_size=32 the largest layer alone requires ~${mb * 32} MB of activation memory.` : `"${top.comp.name}" outputs ${(top.elements / 1e6).toFixed(1)}M elements (~${mb} MB float32/sample). At batch_size=32 this single layer requires ~${mb * 32} MB of activation memory.`,
    affectedIds: offenders.map((o) => o.comp.id),
    suggestion: "Shrink the oversized input dimension (sequence length / spatial size), or add pooling / strided convolutions before the first large layer. Use AMP (float16) to halve activation memory."
  }];
};
var moeNoAuxLoss = (model) => {
  const moes = model.components.filter((c) => c.type === "moeLayer");
  if (moes.length === 0) return [];
  const many = moes.length > 1;
  return [{
    id: `moe-aux-${moes[0].id}`,
    ruleId: "moe-no-aux-loss",
    severity: "info",
    category: "pattern",
    title: many ? `${moes.length} MoE layers: add auxiliary load-balancing loss` : `MoE "${moes[0].name}": add auxiliary load-balancing loss`,
    message: `MoE layers require an auxiliary router z-loss + load-balance loss during training to prevent expert collapse. This is not visible in the architecture diagram but must be in the training loop.${many ? ` Applies to all ${moes.length}: ${moes.map((c) => c.name).join(", ")}.` : ""}`,
    affectedIds: moes.map((c) => c.id),
    suggestion: `Add a note on ${many ? "these layers" : "this layer"}. Typical aux_loss coefficient: 1e-2 (Mixtral/Switch Transformer).`
  }];
};
var gqaHeadMismatch = (model) => {
  return model.components.filter((c) => c.type === "groupedQueryAttention").filter((c) => {
    const H = Number(c.params.numHeads) || 32;
    const Hkv = Number(c.params.numKVHeads) || 8;
    return H % Hkv !== 0;
  }).map((c) => {
    const H = Number(c.params.numHeads) || 32;
    const Hkv = Number(c.params.numKVHeads) || 8;
    const snapped = largestDivisorAtMost(H, Hkv);
    return {
      id: `gqa-mismatch-${c.id}`,
      ruleId: "gqa-head-mismatch",
      severity: "error",
      category: "structure",
      title: `GQA "${c.name}": numHeads not divisible by numKVHeads`,
      message: `numHeads (${H}) must be divisible by numKVHeads (${Hkv}) for grouped-query attention to work correctly.`,
      affectedIds: [c.id],
      suggestion: `Set numKVHeads to a divisor of ${H}, e.g. ${snapped}.`,
      fix: {
        kind: "update-params",
        componentId: c.id,
        params: { numKVHeads: snapped },
        label: `Set numKVHeads to ${snapped}`
      }
    };
  });
};
var swigluDimConvention = (model) => {
  return model.components.filter((c) => c.type === "swiglu").filter((c) => {
    const D = Number(c.params.embedDim ?? c.params.inFeatures) || 4096;
    const I = Number(c.params.intermediateSize ?? c.params.hiddenFeatures ?? c.params.ffDim);
    if (!I) return false;
    const ratio = I / D;
    return ratio < SWIGLU_RATIO_MIN || ratio > SWIGLU_RATIO_MAX;
  }).map((c) => {
    const D = Number(c.params.embedDim ?? c.params.inFeatures) || 4096;
    const I = Number(c.params.intermediateSize ?? c.params.hiddenFeatures ?? c.params.ffDim) || 0;
    const recommended = Math.round(Math.round(D * 8 / 3 / 256) * 256);
    return {
      id: `swiglu-dim-${c.id}`,
      ruleId: "swiglu-dim-convention",
      severity: "info",
      category: "performance",
      title: `SwiGLU "${c.name}": intermediateSize looks non-standard`,
      message: `LLaMA uses intermediateSize \u2248 \u230A(8/3 \xD7 D) / 256\u230B \xD7 256. Current: ${I} (${(I / D).toFixed(2)}\xD7 embedDim). Expected: ~${recommended}.`,
      affectedIds: [c.id],
      suggestion: `Set intermediateSize to ${recommended} for embedDim=${D}.`
    };
  });
};
var CONV_TYPES = /* @__PURE__ */ new Set([
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d"
]);
var linearAfterConvNoFlatten = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (CONV_TYPES.has(from.type) && to.type === "linear") {
      issues.push({
        id: `conv-linear-${conn.id}`,
        ruleId: "linear-after-conv-no-flatten",
        severity: "error",
        category: "structure",
        title: `Conv feeds Linear without flattening: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) outputs a multi-dimensional feature map but connects straight into "${to.name}" (Linear), which expects a flat [batch, features] tensor. At runtime this raises a shape error (or silently mis-multiplies the spatial dims).`,
        affectedIds: [from.id, to.id],
        suggestion: "Insert a Flatten (keep spatial info) or a Global Average Pool (collapse spatial dims) between the convolution and the Linear layer.",
        fix: {
          kind: "insert-on-connection",
          connectionId: conn.id,
          componentType: "flatten",
          label: `Insert Flatten between "${from.name}" and "${to.name}"`
        }
      });
    }
  }
  return issues;
};
var redundantActivation = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (ACTIVATION_TYPES.has(from.type) && ACTIVATION_TYPES.has(to.type)) {
      const same = from.type === to.type;
      issues.push({
        id: `redundant-act-${conn.id}`,
        ruleId: "redundant-activation",
        severity: "warning",
        category: "ordering",
        title: `Back-to-back activations: "${from.name}" \u2192 "${to.name}"`,
        message: same ? `"${from.name}" and "${to.name}" are both ${from.type}. Applying the same activation twice adds compute but no expressivity, almost always a duplicated line.` : `"${from.name}" (${from.type}) feeds straight into "${to.name}" (${to.type}). Stacking two activations with no Linear/Conv between them is rarely intended, e.g. a ReLU before a Softmax clips logits to \u2265 0 and distorts the output distribution.`,
        affectedIds: [from.id, to.id],
        suggestion: "Remove one activation, or insert the Linear/Conv/Norm layer that belongs between them."
      });
    }
  }
  return issues;
};
var consecutiveLinearNoActivation = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (from.type === "linear" && to.type === "linear") {
      issues.push({
        id: `consecutive-linear-${conn.id}`,
        ruleId: "consecutive-linear-no-activation",
        severity: "info",
        category: "pattern",
        title: `Linear \u2192 Linear with no activation: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" feeds directly into "${to.name}" with no activation between them. Two stacked linear maps collapse into one (W\u2082\xB7W\u2081), so the extra layer costs parameters but adds no representational power.`,
        affectedIds: [from.id, to.id],
        suggestion: "Add a non-linearity (ReLU/GELU) between them. If this is a deliberate low-rank / factorized projection (down-proj \u2192 up-proj), this hint is safe to ignore."
      });
    }
  }
  return issues;
};
var dropoutAtOutput = (model) => {
  const outputNode = model.components.find((c) => c.type === "output");
  if (!outputNode) return [];
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections.filter((c) => c.to === outputNode.id)) {
    const from = byId.get(conn.from);
    if (!from || from.type !== "dropout") continue;
    issues.push({
      id: `dropout-output-${conn.id}`,
      ruleId: "dropout-at-output",
      severity: "warning",
      category: "ordering",
      title: `Dropout immediately before Output: "${from.name}"`,
      message: `"${from.name}" (Dropout) is the last layer before Output. In training it randomly zeroes the final logits themselves, injecting noise straight into the loss; at eval it is a no-op, so train and eval behaviour diverge. Dropout belongs before the final projection, not after it.`,
      affectedIds: [from.id, outputNode.id],
      suggestion: "Move the Dropout before the final Linear/Conv that produces the logits."
    });
  }
  return issues;
};
var STRIDED_CONV_TYPES = /* @__PURE__ */ new Set([
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv1d",
  "depthwiseConv2d",
  "separableConv2d"
]);
var RESIDUAL_PASSTHROUGH = /* @__PURE__ */ new Set(["batchNorm", "layerNorm", "groupNorm", "instanceNorm"]);
var ELEMENTWISE_MERGE_TYPES = /* @__PURE__ */ new Set(["add", "multiply", "mean"]);
function isProjectionShortcut(model, convId, kernel) {
  if (kernel !== 1) return false;
  const byId = new Map(model.components.map((c) => [c.id, c]));
  let frontier = [convId];
  for (let hop = 0; hop < 3 && frontier.length; hop++) {
    const next = [];
    for (const id of frontier) {
      for (const e of model.connections.filter((x) => x.from === id)) {
        const to = byId.get(e.to);
        if (!to) continue;
        if (ELEMENTWISE_MERGE_TYPES.has(to.type)) return true;
        if (RESIDUAL_PASSTHROUGH.has(to.type)) next.push(to.id);
      }
    }
    frontier = next;
  }
  return false;
}
var convStrideGtKernel = (model) => {
  return model.components.filter((c) => STRIDED_CONV_TYPES.has(c.type)).map((c) => ({ c, k: Number(c.params.kernelSize), s: Number(c.params.stride) })).filter(({ k, s }) => Number.isFinite(k) && Number.isFinite(s) && k > 0 && s > k).filter(({ c, k }) => !isProjectionShortcut(model, c.id, k)).map(({ c, k, s }) => ({
    id: `conv-stride-gt-kernel-${c.id}`,
    ruleId: "conv-stride-gt-kernel",
    severity: "warning",
    category: "structure",
    title: `Conv stride exceeds kernel: "${c.name}"`,
    message: `"${c.name}" has stride ${s} > kernelSize ${k}. Each step the kernel jumps ${s - k} pixel(s) past its own footprint, so a band of the input is never read, a silent loss of information. Non-overlapping patches use stride == kernel (e.g. ViT 16/16); stride > kernel is almost always a typo.`,
    affectedIds: [c.id],
    suggestion: `Set stride \u2264 kernelSize (${k}); use stride == kernelSize for non-overlapping patches.`
  }));
};
var SPATIAL_CONV_TYPES = /* @__PURE__ */ new Set([
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv1d",
  "depthwiseConv2d",
  "separableConv2d"
]);
var nonSpatialIntoConv = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if ((from.type === "flatten" || from.type === "linear") && SPATIAL_CONV_TYPES.has(to.type)) {
      issues.push({
        id: `non-spatial-into-conv-${conn.id}`,
        ruleId: "non-spatial-into-conv",
        severity: "warning",
        category: "structure",
        title: `Non-spatial tensor into Conv: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) emits a flat [batch, features] vector, but "${to.name}" (${to.type}) expects a [channels, \u2026spatial] feature map. The forward pass raises a shape error unless the dimensions are restored first.`,
        affectedIds: [from.id, to.id],
        suggestion: "Insert a Reshape / Unflatten to rebuild the spatial dims before the convolution."
      });
    }
  }
  return issues;
};
var doubleNorm = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (NORM_TYPES.has(from.type) && NORM_TYPES.has(to.type)) {
      issues.push({
        id: `double-norm-${conn.id}`,
        ruleId: "double-norm",
        severity: "info",
        category: "pattern",
        title: `Back-to-back normalization: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) feeds directly into "${to.name}" (${to.type}). Normalizing an already-normalized tensor is redundant, the second layer mostly re-centres/re-scales what the first produced and just burns its own learnable parameters.`,
        affectedIds: [from.id, to.id],
        suggestion: "Keep a single normalization layer here, or move one of them next to the layer whose activations it should stabilize."
      });
    }
  }
  return issues;
};
var duplicatePositionalEncoding = (model) => {
  const pes = model.components.filter((c) => PE_TYPES.has(c.type));
  if (pes.length < 2) return [];
  const outgoing = /* @__PURE__ */ new Map();
  for (const conn of model.connections) {
    const list = outgoing.get(conn.from);
    if (list) list.push(conn.to);
    else outgoing.set(conn.from, [conn.to]);
  }
  const peById = new Map(pes.map((c) => [c.id, c]));
  for (const pe of pes) {
    const seen = /* @__PURE__ */ new Set([pe.id]);
    const stack = [...outgoing.get(pe.id) ?? []];
    while (stack.length > 0) {
      const id = stack.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      const downstream = peById.get(id);
      if (downstream) {
        return [{
          id: "duplicate-positional-encoding",
          ruleId: "duplicate-positional-encoding",
          severity: "info",
          category: "pattern",
          title: "Position injected twice on one stream",
          message: `"${pe.name}" (${pe.type}) and "${downstream.name}" (${downstream.type}) both add position to the same tensor. Stacking absolute + rotary, or two of the same, double-counts position and tends to hurt more than help. Separate towers with one each (a vision and a text branch, an encoder and a decoder) are not this.`,
          affectedIds: [pe.id, downstream.id],
          suggestion: "Keep a single positional scheme on this path: sinusoidal OR learned OR RoPE / ALiBi."
        }];
      }
      stack.push(...outgoing.get(id) ?? []);
    }
  }
  return [];
};
var SPATIAL_POOL_TYPES = /* @__PURE__ */ new Set([
  "maxpool2d",
  "avgpool2d",
  "maxpool1d",
  "avgpool1d",
  "adaptiveMaxPool2d",
  "adaptiveAvgPool2d"
]);
var poolIntoLinearNoFlatten = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (SPATIAL_POOL_TYPES.has(from.type) && to.type === "linear") {
      issues.push({
        id: `pool-into-linear-${conn.id}`,
        ruleId: "pool-into-linear-no-flatten",
        severity: "warning",
        category: "structure",
        title: `Pooling feeds Linear without flattening: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (${from.type}) still carries [channels, \u2026spatial] dims, but "${to.name}" (Linear) expects a flat [batch, features] tensor. At runtime this raises a shape error.`,
        affectedIds: [from.id, to.id],
        suggestion: "Insert a Flatten (keep spatial info) or a Global Average Pool (collapse spatial dims) between the pooling and the Linear layer.",
        fix: {
          kind: "insert-on-connection",
          connectionId: conn.id,
          componentType: "flatten",
          label: `Insert Flatten between "${from.name}" and "${to.name}"`
        }
      });
    }
  }
  return issues;
};
var flattenIntoAttention = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const issues = [];
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (from.type === "flatten" && ATTENTION_TYPES.has(to.type)) {
      issues.push({
        id: `flatten-into-attention-${conn.id}`,
        ruleId: "flatten-into-attention",
        severity: "warning",
        category: "structure",
        title: `Flatten before attention: "${from.name}" \u2192 "${to.name}"`,
        message: `"${from.name}" (Flatten) collapses the sequence dimension into one long vector, but "${to.name}" (${to.type}) attends over a [sequence, dim] tensor. Flattening first leaves a length-1 sequence, so attention has nothing to relate.`,
        affectedIds: [from.id, to.id],
        suggestion: "Remove the Flatten before attention; keep the [sequence, dim] layout and flatten (or pool) only after the attention stack."
      });
    }
  }
  return issues;
};
var transposeConvCheckerboard = (model) => {
  return model.components.filter((c) => c.type === "transposeConv2d").map((c) => ({ c, k: Number(c.params.kernelSize), s: Number(c.params.stride) })).filter(({ k, s }) => Number.isFinite(k) && Number.isFinite(s) && s > 1 && k % s !== 0).map(({ c, k, s }) => ({
    id: `transposeconv-checkerboard-${c.id}`,
    ruleId: "transposeconv-checkerboard",
    severity: "info",
    category: "pattern",
    title: `Checkerboard risk: "${c.name}"`,
    message: `ConvTranspose "${c.name}" has kernelSize ${k} not divisible by stride ${s}. The uneven kernel overlap during upsampling deposits more weight on some output pixels than others, producing checkerboard artifacts.`,
    affectedIds: [c.id],
    suggestion: `Make kernelSize a multiple of stride (e.g. ${s * Math.max(2, Math.round(k / s))}), or upsample with Upsample + Conv (resize-convolution) instead.`
  }));
};
var groupNormDivisibility = (model) => {
  const issues = [];
  for (const c of model.components) {
    if (c.type !== "groupNorm") continue;
    const groups = Number(c.params.numGroups);
    const channels = Number(
      c.params.numChannels ?? (Array.isArray(c.inputShape) ? c.inputShape[0] : void 0)
    );
    if (!Number.isFinite(groups) || !Number.isFinite(channels) || groups <= 0 || channels <= 0) continue;
    if (channels % groups !== 0) {
      const divisor = [1, 2, 4, 8, 16, 32].filter((d) => channels % d === 0).pop() ?? 1;
      issues.push({
        id: `groupnorm-divisibility-${c.id}`,
        ruleId: "groupnorm-divisibility",
        severity: "error",
        category: "structure",
        title: `GroupNorm "${c.name}": channels not divisible by numGroups`,
        message: `GroupNorm "${c.name}" has numGroups=${groups} but ${channels} channels. ${channels} % ${groups} = ${channels % groups}; PyTorch requires the channel count to be an exact multiple of numGroups.`,
        affectedIds: [c.id],
        suggestion: `Set numGroups to a divisor of ${channels}, e.g. ${divisor}.`,
        fix: {
          kind: "update-params",
          componentId: c.id,
          params: { numGroups: divisor },
          label: `Set numGroups to ${divisor}`
        }
      });
    }
  }
  return issues;
};
var hugeLinearParams = (model) => {
  const issues = [];
  for (const c of model.components) {
    if (c.type !== "linear") continue;
    const outF = Number(c.params.outFeatures);
    const inF = Number(
      c.params.inFeatures ?? (Array.isArray(c.inputShape) ? c.inputShape[c.inputShape.length - 1] : void 0)
    );
    if (!Number.isFinite(inF) || !Number.isFinite(outF) || inF <= 0 || outF <= 0) continue;
    const count = inF * outF;
    if (count > LARGE_LINEAR_MAX_PARAMS) {
      const gb = (count * 4 / 1073741824).toFixed(1);
      issues.push({
        id: `huge-linear-${c.id}`,
        ruleId: "huge-linear-params",
        severity: "warning",
        category: "performance",
        title: `Very large Linear: "${c.name}" (~${Math.round(count / 1e6)}M params)`,
        message: `Linear "${c.name}" is ${inF} \xD7 ${outF} = ${Math.round(count / 1e6)}M parameters (~${gb} GB float32). A single dense layer this large usually means a feature map was flattened without pooling first; embedding / vocab-projection heads are the expected exception.`,
        affectedIds: [c.id],
        suggestion: "Add a Global Average Pool or more downsampling before the Linear, or factorize it (low-rank / bottleneck projection)."
      });
    }
  }
  return issues;
};
var MHA_SERVING_MIN_LAYERS = 6;
var MHA_SERVING_MIN_DIM = 2048;
var MHA_FAMILY = /* @__PURE__ */ new Set(["multiHeadAttention", "selfAttention", "causalAttention"]);
var fullMhaServingCost = (model) => {
  if (model.components.some((c) => c.type === "mla")) return [];
  const reducedGqa = model.components.some(
    (c) => c.type === "groupedQueryAttention" && Number(c.params.numKVHeads) > 0 && Number(c.params.numKVHeads) < Number(c.params.numHeads ?? c.params.numKVHeads)
  );
  if (reducedGqa) return [];
  const mhaLayers = model.components.filter((c) => {
    if (MHA_FAMILY.has(c.type)) return true;
    if (c.type === "groupedQueryAttention") {
      const q = Number(c.params.numHeads);
      const kv = Number(c.params.numKVHeads);
      return !(kv > 0 && q > 0 && kv < q);
    }
    return false;
  });
  if (mhaLayers.length < MHA_SERVING_MIN_LAYERS) return [];
  const dims = mhaLayers.map((c) => Number(c.params.embedDim ?? c.params.hiddenDim ?? c.params.dModel)).filter((d) => Number.isFinite(d) && d > 0);
  if (dims.length === 0) return [];
  const embedDim = Math.round(dims.reduce((a, b) => a + b, 0) / dims.length);
  if (embedDim < MHA_SERVING_MIN_DIM) return [];
  const kvPerTokenKB = mhaLayers.length * 2 * embedDim * 2 / 1024;
  return [{
    id: "full-mha-serving",
    ruleId: "full-mha-serving-cost",
    severity: "info",
    category: "performance",
    title: `Full multi-head attention at LLM scale (~${kvPerTokenKB.toFixed(0)} KB/token KV cache)`,
    message: `${mhaLayers.length} attention layers at embedDim ${embedDim} cache full per-head K/V: about ${kvPerTokenKB.toFixed(0)} KB per token at fp16, which dominates memory at long context. Grouped-query attention (e.g. 8:1) would cut this ~8\xD7; multi-head latent attention (MLA) shrinks it ~10\xD7 or more. This is the move production LLMs make; it does not change the parameter count.`,
    affectedIds: mhaLayers.map((c) => c.id),
    suggestion: "Switch attention to groupedQueryAttention (set numKVHeads below numHeads, e.g. numHeads/4) or mla (a low-rank cached latent)."
  }];
};
var SATURATING_ACTIVATIONS = /* @__PURE__ */ new Set(["sigmoid", "tanh"]);
var WEIGHTED_INIT_TYPES = /* @__PURE__ */ new Set([
  "linear",
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwiseConv2d",
  "separableConv2d",
  "transposeConv2d"
]);
var initActivationMismatch = (model) => {
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const affected = [];
  const actNames = /* @__PURE__ */ new Set();
  for (const conn of model.connections) {
    const from = byId.get(conn.from);
    const to = byId.get(conn.to);
    if (!from || !to) continue;
    if (WEIGHTED_INIT_TYPES.has(from.type) && SATURATING_ACTIVATIONS.has(to.type)) {
      affected.push(from.id, to.id);
      actNames.add(to.type);
    }
  }
  if (affected.length === 0) return [];
  return [{
    id: "init-activation-mismatch",
    ruleId: "init-activation-mismatch",
    severity: "info",
    category: "pattern",
    title: `Default init assumes ReLU, but ${[...actNames].join("/")} follows`,
    message: "PyTorch initializes Linear/Conv with Kaiming (He) init, which is derived for ReLU-family activations. Feeding a saturating activation (sigmoid/tanh) from a He-initialized layer starts training in the saturated tails, shrinking early gradients.",
    affectedIds: [...new Set(affected)],
    suggestion: 'Initialize these layers with Xavier instead: nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("sigmoid"|"tanh")), or switch the activation to a ReLU-family one.'
  }];
};
var deepAttentionDefaultInit = (model) => {
  const attn = model.components.filter((c) => ATTENTION_TYPES.has(c.type));
  if (attn.length < SCALED_INIT_MIN_ATTENTION_LAYERS) return [];
  const advice = getInitAdvice("causalAttention");
  return [{
    id: "deep-attention-default-init",
    ruleId: "deep-attention-default-init",
    severity: "info",
    category: "pattern",
    title: `${attn.length} attention layers: use depth-scaled init`,
    message: `At ${attn.length} stacked attention layers, residual-branch outputs add up; unscaled init lets activation variance grow with depth. GPT-2/LLaMA-family models scale the residual projections by depth${advice ? ` (${advice.formula})` : ""}.`,
    affectedIds: attn.map((c) => c.id),
    suggestion: advice ? `Scale residual output projections by depth: ${advice.pyTorch}` : "Scale residual output projections by 1/sqrt(2 \xB7 numLayers)."
  }];
};
var lmHeadVocabMismatch = (model) => {
  const vocabs = model.components.filter((c) => c.type === "embedding").map((c) => Number(c.params.vocabSize)).filter((v) => Number.isFinite(v) && v > 0);
  if (vocabs.length === 0) return [];
  if (!model.components.some((c) => ATTENTION_TYPES.has(c.type))) return [];
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const outputIds = new Set(model.components.filter((c) => c.type === "output").map((c) => c.id));
  if (outputIds.size === 0) return [];
  const issues = [];
  for (const conn of model.connections) {
    if (!outputIds.has(conn.to)) continue;
    const head = byId.get(conn.from);
    if (!head || head.type !== "linear") continue;
    const out = Number(head.params.outFeatures);
    if (!Number.isFinite(out) || out <= 0) continue;
    if (vocabs.includes(out)) continue;
    issues.push({
      id: `lm-head-vocab-${head.id}`,
      ruleId: "lm-head-vocab-mismatch",
      severity: "info",
      category: "structure",
      title: `Head projects to ${out}, embedding vocab is ${vocabs[0]}`,
      message: `"${head.name}" feeds the output with outFeatures ${out}, but the embedding vocabulary is ${vocabs.join("/")}. For a language model the head must project back to vocab size (and is usually weight-tied to the embedding). If this is a classifier head over ${out} classes, ignore.`,
      affectedIds: [head.id],
      suggestion: `If this model predicts tokens, set outFeatures to ${vocabs[0]} and consider tying the head to the embedding weights.`
    });
  }
  return issues;
};
var kvCacheContextBudget = (model) => {
  let perTokenBytes = 0;
  const affected = [];
  for (const c of model.components) {
    if (!ATTENTION_TYPES.has(c.type) || c.type === "mla") continue;
    const embedDim = Number(c.params.embedDim ?? c.params.hiddenDim ?? c.params.dModel);
    const numHeads = Number(c.params.numHeads);
    if (!Number.isFinite(embedDim) || embedDim <= 0) continue;
    const kvHeads = c.type === "groupedQueryAttention" && Number(c.params.numKVHeads) > 0 && Number(c.params.numKVHeads) < numHeads ? Number(c.params.numKVHeads) : numHeads;
    const headDim = Number.isFinite(numHeads) && numHeads > 0 ? embedDim / numHeads : embedDim;
    const kvWidth = Number.isFinite(kvHeads) && kvHeads > 0 ? kvHeads * headDim : embedDim;
    perTokenBytes += 2 * kvWidth * 2;
    affected.push(c.id);
  }
  if (perTokenBytes === 0) return [];
  const totalGB = perTokenBytes * KV_BUDGET_CONTEXT_TOKENS / 1e9;
  if (totalGB <= KV_BUDGET_MAX_GB) return [];
  return [{
    id: "kv-cache-context-budget",
    ruleId: "kv-cache-context-budget",
    severity: "warning",
    category: "performance",
    title: `KV cache \u2248 ${totalGB.toFixed(1)} GB at ${KV_BUDGET_CONTEXT_TOKENS.toLocaleString()} tokens (fp16, 1 sequence)`,
    message: `Across ${affected.length} attention layers this design caches ${(perTokenBytes / 1024).toFixed(0)} KB per token, so a single ${KV_BUDGET_CONTEXT_TOKENS.toLocaleString()}-token sequence needs ~${totalGB.toFixed(1)} GB of KV cache before weights or activations. That exceeds the ${KV_BUDGET_MAX_GB} GB budget this rule assumes for serving headroom.`,
    affectedIds: affected,
    suggestion: "Cut KV width: raise the GQA ratio (fewer numKVHeads), switch to MLA, reduce depth or embedDim, or accept a shorter serving context."
  }];
};
var ALL_RULES = [
  noInputNode,
  // R01 structure  error
  noOutputNode,
  // R02 structure  warning
  isolatedComponents,
  // R03 structure  warning
  deadEnds,
  // R04 structure  warning
  bnAfterActivation,
  // R05 ordering   warning
  dropoutBeforeBN,
  // R06 ordering   info
  outputActivation,
  // R07 ordering   info
  bnAtOutput,
  // R08 ordering   warning
  deepNoResidual,
  // R09 pattern    warning
  attentionNoPE,
  // R10 pattern    warning
  vanishingGradientRisk,
  // R11 pattern info
  deepNoNorm,
  // R12 pattern    info
  highDropout,
  // R13 performance warning
  largeActivation,
  // R14 performance warning
  moeNoAuxLoss,
  // R15 pattern    info
  gqaHeadMismatch,
  // R16 structure  error
  swigluDimConvention,
  // R17 performance info
  linearAfterConvNoFlatten,
  // R18 structure error
  redundantActivation,
  // R19 ordering   warning
  consecutiveLinearNoActivation,
  // R20 pattern info
  dropoutAtOutput,
  // R21 ordering   warning
  convStrideGtKernel,
  // R22 structure  warning
  nonSpatialIntoConv,
  // R23 structure  warning
  doubleNorm,
  // R24 pattern    info
  duplicatePositionalEncoding,
  // R25 pattern info
  poolIntoLinearNoFlatten,
  // R26 structure  warning
  flattenIntoAttention,
  // R27 structure  warning
  transposeConvCheckerboard,
  // R28 pattern   info
  groupNormDivisibility,
  // R29 structure  error
  hugeLinearParams,
  // R30 performance warning
  fullMhaServingCost,
  // R31 performance info
  initActivationMismatch,
  // R32 pattern    info
  deepAttentionDefaultInit,
  // R33 pattern    info
  lmHeadVocabMismatch,
  // R34 structure  info
  kvCacheContextBudget
  // R35 performance warning
];
var ADVISOR_RULE_IDS = [
  "no-input-node",
  "no-output-node",
  "isolated-components",
  "dead-end",
  "bn-after-activation",
  "dropout-before-bn",
  "output-activation",
  "bn-at-output",
  "deep-no-residual",
  "attention-no-pe",
  "vanishing-gradient",
  "deep-no-norm",
  "high-dropout",
  "large-activation",
  "moe-no-aux-loss",
  "gqa-head-mismatch",
  "swiglu-dim-convention",
  "linear-after-conv-no-flatten",
  "redundant-activation",
  "consecutive-linear-no-activation",
  "dropout-at-output",
  "conv-stride-gt-kernel",
  "non-spatial-into-conv",
  "double-norm",
  "duplicate-positional-encoding",
  "pool-into-linear-no-flatten",
  "flatten-into-attention",
  "transposeconv-checkerboard",
  "groupnorm-divisibility",
  "huge-linear-params",
  "full-mha-serving-cost",
  "init-activation-mismatch",
  "deep-attention-default-init",
  "lm-head-vocab-mismatch",
  "kv-cache-context-budget"
];
function runAdvisorRules(model, opts = {}) {
  if (model.components.length === 0) return [];
  const issues = [];
  for (const rule of ALL_RULES) {
    try {
      issues.push(...rule(model));
    } catch (e) {
      console.warn("[ArchitectureAdvisor] Rule error:", e);
    }
  }
  if (!opts.excludeIssueIds?.length) return issues;
  const drop = new Set(opts.excludeIssueIds);
  return issues.filter((i) => !drop.has(i.id));
}

// src/utils/lintEngine.ts
init_paramEstimator();

// src/utils/hardwareSpecs.ts
var GPU_HARDWARE = [
  { id: "b200", label: "B200 (192 GB)", tier: "datacenter", memoryGB: 192, bf16TFLOPS: 2250, fp32TFLOPS: 80, memBandwidthGBs: 8e3, nvlinkGBs: 900, nodeEgressGBs: 800 },
  { id: "h200", label: "H200 (141 GB)", tier: "datacenter", memoryGB: 141, bf16TFLOPS: 989, fp32TFLOPS: 67, memBandwidthGBs: 4800, nvlinkGBs: 450, nodeEgressGBs: 400 },
  { id: "h100-sxm", label: "H100 SXM (80 GB)", tier: "datacenter", memoryGB: 80, bf16TFLOPS: 989, fp32TFLOPS: 67, memBandwidthGBs: 3350, nvlinkGBs: 450, nodeEgressGBs: 400 },
  { id: "a100-80g", label: "A100 (80 GB)", tier: "datacenter", memoryGB: 80, bf16TFLOPS: 312, fp32TFLOPS: 19.5, memBandwidthGBs: 2039, nvlinkGBs: 300, nodeEgressGBs: 200 },
  { id: "a100-40g", label: "A100 (40 GB)", tier: "datacenter", memoryGB: 40, bf16TFLOPS: 312, fp32TFLOPS: 19.5, memBandwidthGBs: 1555, nvlinkGBs: 300, nodeEgressGBs: 200 },
  { id: "l40s", label: "L40S (48 GB)", tier: "datacenter", memoryGB: 48, bf16TFLOPS: 181, fp32TFLOPS: 91.6, memBandwidthGBs: 864, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "l4", label: "L4 (24 GB)", tier: "datacenter", memoryGB: 24, bf16TFLOPS: 121, fp32TFLOPS: 30.3, memBandwidthGBs: 300, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "a10g", label: "A10G (24 GB)", tier: "datacenter", memoryGB: 24, bf16TFLOPS: 70, fp32TFLOPS: 31.2, memBandwidthGBs: 600, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "t4", label: "T4 (16 GB)", tier: "datacenter", memoryGB: 16, bf16TFLOPS: 65, fp32TFLOPS: 8.1, memBandwidthGBs: 320, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "v100", label: "V100 (16 GB)", tier: "datacenter", memoryGB: 16, bf16TFLOPS: 125, fp32TFLOPS: 15.7, memBandwidthGBs: 900, nvlinkGBs: 150, nodeEgressGBs: 100 },
  // Consumer bf16 numbers use fp32 accumulate (the training-relevant rate).
  { id: "rtx-4090", label: "RTX 4090 (24 GB)", tier: "consumer", memoryGB: 24, bf16TFLOPS: 82.6, fp32TFLOPS: 82.6, memBandwidthGBs: 1008, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "rtx-3090", label: "RTX 3090 (24 GB)", tier: "consumer", memoryGB: 24, bf16TFLOPS: 71, fp32TFLOPS: 35.6, memBandwidthGBs: 936, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "m3-max", label: "M3 Max (128 GB)", tier: "apple", memoryGB: 128, bf16TFLOPS: 14.2, fp32TFLOPS: 14.2, memBandwidthGBs: 400, nvlinkGBs: 0, nodeEgressGBs: 0 },
  { id: "m3-pro", label: "M3 Pro (36 GB)", tier: "apple", memoryGB: 36, bf16TFLOPS: 6, fp32TFLOPS: 6, memBandwidthGBs: 150, nvlinkGBs: 0, nodeEgressGBs: 0 },
  // TPUs (scaling-book ch. 2 numbers). nvlinkGBs carries the per-chip ICI
  // egress; TPU pods route everything over ICI so nodeEgressGBs stays 0.
  // fp32TFLOPS mirrors bf16 because no consumer routes TPUs through an fp32
  // path (training panels filter the tier out); do not read it as MXU fp32.
  { id: "tpu-v5e", label: "TPU v5e (16 GB)", tier: "tpu", memoryGB: 16, bf16TFLOPS: 197, fp32TFLOPS: 197, memBandwidthGBs: 819, nvlinkGBs: 180, nodeEgressGBs: 0 },
  { id: "tpu-v5p", label: "TPU v5p (96 GB)", tier: "tpu", memoryGB: 96, bf16TFLOPS: 459, fp32TFLOPS: 459, memBandwidthGBs: 2765, nvlinkGBs: 540, nodeEgressGBs: 0 },
  { id: "tpu-v6e", label: "TPU v6e (32 GB)", tier: "tpu", memoryGB: 32, bf16TFLOPS: 918, fp32TFLOPS: 918, memBandwidthGBs: 1640, nvlinkGBs: 360, nodeEgressGBs: 0 }
];
var BY_ID = new Map(GPU_HARDWARE.map((g) => [g.id, g]));

// src/utils/costEstimator.ts
var GPU_SPECS = GPU_HARDWARE.filter((g) => g.tier !== "apple" && g.tier !== "tpu").map((g) => ({
  id: g.id,
  label: g.label,
  memoryGB: g.memoryGB,
  peakTFLOPS: g.bf16TFLOPS,
  peakFP32TFLOPS: g.fp32TFLOPS,
  memBandwidthGBs: g.memBandwidthGBs
}));

// src/utils/shapeInference.ts
var ELEMENTWISE_MERGE = /* @__PURE__ */ new Set(["add", "multiply", "mean"]);
var CONCAT_MERGE = /* @__PURE__ */ new Set(["concatenate"]);
var MATMUL_LAYERS = /* @__PURE__ */ new Set(["matmul"]);
var EMBED_MATCH_LAYERS = /* @__PURE__ */ new Set(["crossAttention"]);
var HEAD_DIM_LAYERS = /* @__PURE__ */ new Set([
  "multiHeadAttention",
  "selfAttention",
  "crossAttention",
  "attention",
  "causalAttention",
  "localAttention",
  "linearAttention",
  "groupedQueryAttention",
  // Further attention variants that carry the same embedDim(/dModel) % numHeads
  // contract (verified they expose both params in the registry).
  "crossModalAttention",
  "coAttention",
  "windowAttention",
  "differentialAttention",
  "nativeSparseAttention",
  "deformableAttention",
  "dividedSpaceTimeAttention",
  "gatedLinearAttention",
  // Block-level types carry the same embedDim/numHeads contract as their
  // inner attention; a non-divisible pair crashes identically at runtime.
  "transformerBlock",
  "conformerBlock",
  "ditBlock",
  "mla"
]);
function isFiniteDim(n) {
  return typeof n === "number" && Number.isFinite(n);
}
function shapeIsInvalid(s) {
  if (!s || s.length === 0) return false;
  return s.some((d) => !isFiniteDim(d) || d <= 0);
}
function shapesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function topoSort(model) {
  const indeg = /* @__PURE__ */ new Map();
  const adj = /* @__PURE__ */ new Map();
  for (const c of model.components) {
    indeg.set(c.id, 0);
    adj.set(c.id, []);
  }
  for (const conn of model.connections) {
    if (!indeg.has(conn.from) || !indeg.has(conn.to)) continue;
    adj.get(conn.from).push(conn.to);
    indeg.set(conn.to, (indeg.get(conn.to) ?? 0) + 1);
  }
  const queue = [];
  for (const [id, d] of indeg) if (d === 0) queue.push(id);
  const order = [];
  while (queue.length) {
    const id = queue.shift();
    order.push(id);
    for (const nb of adj.get(id) ?? []) {
      const d = (indeg.get(nb) ?? 0) - 1;
      indeg.set(nb, d);
      if (d === 0) queue.push(nb);
    }
  }
  return order;
}
function checkAttentionDivisibility(c) {
  const p = c.params ?? {};
  const embed = p.embedDim ?? p.hiddenDim ?? p.dModel;
  const heads = p.numHeads;
  if (typeof embed !== "number" || typeof heads !== "number") return null;
  if (heads <= 0) return null;
  if (embed % heads !== 0) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "head-dim-divisibility",
      message: `${c.type} '${c.name}': embed dim (${embed}) must be divisible by numHeads (${heads}); head_dim would be ${(embed / heads).toFixed(2)}`,
      expected: 0,
      actual: embed % heads
    };
  }
  if (c.type === "groupedQueryAttention") {
    const kvHeads = p.numKVHeads;
    if (typeof kvHeads === "number" && kvHeads > 0 && heads % kvHeads !== 0) {
      return {
        componentId: c.id,
        componentName: c.name,
        componentType: c.type,
        kind: "gqa-head-divisibility",
        message: `groupedQueryAttention '${c.name}': numHeads (${heads}) must be divisible by numKVHeads (${kvHeads})`,
        expected: 0,
        actual: heads % kvHeads
      };
    }
  }
  return null;
}
function checkAttentionInDim(c, inShape) {
  const p = c.params ?? {};
  const embed = p.embedDim ?? p.hiddenDim ?? p.dModel;
  if (typeof embed !== "number" || inShape.length === 0) return null;
  const lastDim = inShape[inShape.length - 1];
  if (!isFiniteDim(lastDim) || lastDim === embed) return null;
  return {
    componentId: c.id,
    componentName: c.name,
    componentType: c.type,
    kind: "attention-in-mismatch",
    message: `${c.type} '${c.name}': embedDim=${embed} but upstream last dim is ${lastDim}; attention expects them equal (project the input or fix embedDim)`,
    expected: embed,
    actual: lastDim
  };
}
function checkLinearInFeatures(c, inShape) {
  if (c.type !== "linear") return null;
  const inFeatures = (c.params ?? {}).inFeatures;
  if (typeof inFeatures !== "number" || !inShape || inShape.length === 0) return null;
  const lastDim = inShape[inShape.length - 1];
  if (lastDim !== inFeatures) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "linear-in-mismatch",
      message: `linear '${c.name}': inFeatures=${inFeatures} but upstream last dim is ${lastDim}`,
      expected: inFeatures,
      actual: lastDim
    };
  }
  return null;
}
function checkMergeShapes(c, parentShapes) {
  if (parentShapes.length < 2) return null;
  if (ELEMENTWISE_MERGE.has(c.type)) {
    const first = parentShapes[0];
    for (let i = 1; i < parentShapes.length; i++) {
      if (!shapesEqual(first, parentShapes[i])) {
        return {
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "merge-shape-mismatch",
          message: `${c.type} '${c.name}': elementwise inputs must match, got [${first.join(",")}] vs [${parentShapes[i].join(",")}]`,
          expected: first,
          actual: parentShapes[i]
        };
      }
    }
    return null;
  }
  if (CONCAT_MERGE.has(c.type)) {
    const p = c.params ?? {};
    const rank = parentShapes[0].length;
    const dimRaw = p.dim ?? -1;
    const dim = dimRaw < 0 ? rank + dimRaw : dimRaw;
    for (let i = 1; i < parentShapes.length; i++) {
      const a = parentShapes[0];
      const b = parentShapes[i];
      if (a.length !== b.length) {
        return {
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "merge-shape-mismatch",
          message: `concat '${c.name}': inputs must have same rank, got rank ${a.length} vs ${b.length}`
        };
      }
      for (let k = 0; k < a.length; k++) {
        if (k === dim) continue;
        if (a[k] !== b[k]) {
          return {
            componentId: c.id,
            componentName: c.name,
            componentType: c.type,
            kind: "merge-shape-mismatch",
            message: `concat '${c.name}' on dim ${dim}: non-concat dim ${k} differs, ${a[k]} vs ${b[k]}`,
            expected: a,
            actual: b
          };
        }
      }
    }
    return null;
  }
  return null;
}
function checkMatmulShapes(c, parentShapes) {
  if (!MATMUL_LAYERS.has(c.type)) return null;
  if (parentShapes.length < 2) return null;
  const a = parentShapes[0];
  const b = parentShapes[1];
  if (a.length < 2 || b.length < 2) return null;
  const inner = a[a.length - 1];
  const other = b[b.length - 2];
  if (inner !== other) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "matmul-shape-mismatch",
      message: `matmul '${c.name}': inner dims must match: first operand's last dim ${inner} vs second operand's dim ${other} ([${a.join(",")}] @ [${b.join(",")}])`,
      expected: a,
      actual: b
    };
  }
  return null;
}
function checkEmbedMatchShapes(c, parentShapes) {
  if (!EMBED_MATCH_LAYERS.has(c.type)) return null;
  if (parentShapes.length < 2) return null;
  const widths = parentShapes.map((s) => s[s.length - 1]);
  const first = widths[0];
  const mismatch = widths.find((w) => w !== first);
  if (mismatch !== void 0) {
    return {
      componentId: c.id,
      componentName: c.name,
      componentType: c.type,
      kind: "cross-attention-kv-mismatch",
      message: `${c.type} '${c.name}': query and key/value streams must share feature width, got ${widths.join(" vs ")}`,
      expected: parentShapes[0],
      actual: parentShapes[1]
    };
  }
  return null;
}
function propagateShapes(model) {
  const order = topoSort(model);
  const shapes = /* @__PURE__ */ new Map();
  const issues = [];
  const compById = new Map(model.components.map((c) => [c.id, c]));
  const parents = /* @__PURE__ */ new Map();
  for (const c of model.components) parents.set(c.id, []);
  for (const conn of model.connections) parents.get(conn.to)?.push(conn.from);
  for (const id of order) {
    const c = compById.get(id);
    if (!c) continue;
    const parentIds = parents.get(id) ?? [];
    const parentOuts = parentIds.map((p) => shapes.get(p)?.out ?? null).filter((s) => s !== null);
    if (HEAD_DIM_LAYERS.has(c.type)) {
      const att = checkAttentionDivisibility(c);
      if (att) issues.push(att);
    }
    const merge = checkMergeShapes(c, parentOuts);
    if (merge) issues.push(merge);
    const mm = checkMatmulShapes(c, parentOuts);
    if (mm) issues.push(mm);
    const emb = checkEmbedMatchShapes(c, parentOuts);
    if (emb) issues.push(emb);
    let inShape = null;
    if (c.type === "input") {
      const s = (c.params ?? {}).shape;
      if (Array.isArray(s) && s.length > 0 && s.every(isFiniteDim)) {
        inShape = s;
      } else {
        issues.push({
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "no-input-shape",
          message: `input '${c.name}': params.shape is not set; downstream shapes cannot be inferred`
        });
      }
    } else if (parentOuts.length >= 1) {
      inShape = parentOuts[0];
    }
    if (c.type === "linear" && inShape) {
      const lin = checkLinearInFeatures(c, inShape);
      if (lin) issues.push(lin);
    }
    if (HEAD_DIM_LAYERS.has(c.type) && inShape) {
      const dim = checkAttentionInDim(c, inShape);
      if (dim) issues.push(dim);
    }
    let outShape = null;
    if (inShape) {
      try {
        const def = componentRegistry[c.type];
        if (def) {
          outShape = def.computeOutputShape(
            inShape,
            c.params ?? {},
            parentOuts.length > 1 ? parentOuts : void 0
          );
        } else {
          issues.push({
            componentId: c.id,
            componentName: c.name,
            componentType: c.type,
            kind: "unknown-layer-type",
            message: `'${c.name}': "${c.type}" is not a known layer type; nothing downstream of it can be verified and the trainer would drop it`
          });
        }
      } catch (err) {
        issues.push({
          componentId: c.id,
          componentName: c.name,
          componentType: c.type,
          kind: "compute-error",
          message: `${c.type} '${c.name}': computeOutputShape threw ${err instanceof Error ? err.message : String(err)}`
        });
      }
    } else if (c.type === "input") {
      const s = (c.params ?? {}).shape;
      if (Array.isArray(s) && s.length > 0 && s.every(isFiniteDim)) {
        outShape = s;
      }
    }
    if (outShape && shapeIsInvalid(outShape) && inShape && !shapeIsInvalid(inShape)) {
      issues.push({
        componentId: c.id,
        componentName: c.name,
        componentType: c.type,
        kind: "invalid-output-shape",
        message: `${c.type} '${c.name}': computed output shape [${outShape.join(",")}] contains a non-positive or non-finite dim`,
        actual: outShape
      });
    }
    shapes.set(id, { in: inShape, out: outShape });
  }
  return { shapes, issues };
}

// src/utils/lintGraph.ts
function advisorSeverity(sev) {
  return sev === "error" ? "block" : sev === "warning" ? "warn" : "info";
}
var SHAPE_SEVERITY = {
  "head-dim-divisibility": "block",
  "gqa-head-divisibility": "block",
  "merge-shape-mismatch": "block",
  "compute-error": "block",
  // A type the registry doesn't know halts propagation, making everything
  // downstream unverifiable, and the trainer codegen drops the layer. Grounded
  // run 2026-08-20: such a design trained to below-random accuracy with a
  // clean static verdict. Unverifiable must never read as fine.
  "unknown-layer-type": "block",
  "invalid-output-shape": "warn",
  "attention-in-mismatch": "warn"
};
var SHAPE_RULE_IDS = Object.keys(SHAPE_SEVERITY);
var ALL_RULE_IDS = [
  .../* @__PURE__ */ new Set([...ADVISOR_RULE_IDS, ...SHAPE_RULE_IDS])
];
function dedupe(findings) {
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  for (const f of findings) {
    const key = `${f.rule}::${f.componentName ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(f);
  }
  return out;
}
function lintModelGraph(model) {
  if (!model || !Array.isArray(model.components)) return [];
  const usable = model.components.filter(
    (c) => !!c && typeof c === "object"
  );
  if (usable.length !== model.components.length) {
    model = { ...model, components: usable };
  }
  const byId = new Map(model.components.map((c) => [c.id, c]));
  const findings = [];
  for (const issue of runAdvisorRules(model)) {
    const target = issue.affectedIds.map((id) => byId.get(id)).find(Boolean);
    findings.push({
      rule: issue.ruleId,
      severity: advisorSeverity(issue.severity),
      message: issue.suggestion ? `${issue.message} Fix: ${issue.suggestion}` : issue.message,
      componentName: target?.name,
      componentType: target?.type
    });
  }
  for (const issue of propagateShapes(model).issues) {
    const severity = SHAPE_SEVERITY[issue.kind];
    if (!severity) continue;
    findings.push({
      rule: issue.kind,
      severity,
      message: issue.message,
      componentName: issue.componentName,
      componentType: issue.componentType
    });
  }
  return dedupe(findings);
}

// src/utils/lintEngine.ts
function graphFromPyTorchSource(code, name = "model") {
  try {
    const parsed = parsePyTorchCode(code);
    if (!parsed || parsed.components.length === 0) return null;
    const realLayers = parsed.components.filter((c) => c.type !== "input" && c.type !== "output");
    if (realLayers.length === 0) return null;
    return stabilizeIds({
      id: name,
      name,
      components: parsed.components,
      connections: parsed.connections
    });
  } catch {
    return null;
  }
}
function stabilizeIds(model) {
  const remap = /* @__PURE__ */ new Map();
  const perType = /* @__PURE__ */ new Map();
  for (const c of model.components) {
    const n = (perType.get(c.type) ?? 0) + 1;
    perType.set(c.type, n);
    remap.set(c.id, `${c.type}-${n}`);
  }
  const to = (id) => remap.get(id) ?? id;
  return {
    ...model,
    components: model.components.map((c) => ({
      ...c,
      id: to(c.id),
      inputs: (c.inputs ?? []).map(to),
      outputs: (c.outputs ?? []).map(to)
    })),
    connections: model.connections.map((e, i) => ({
      ...e,
      id: `e-${i + 1}`,
      from: to(e.from),
      to: to(e.to)
    }))
  };
}

// src/utils/customLayerStore.ts
var STORAGE_KEY = "neurarch-custom-layers";
function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
function getCustomLayerById(id) {
  return load().find((l) => l.id === id);
}

// src/utils/codeGenerator.ts
function pyDim(v, fallback) {
  if (Array.isArray(v)) return `(${v.map((n) => typeof n === "number" ? n : Number(n) || 0).join(", ")})`;
  if (v === void 0 || v === null || v === "") return String(fallback);
  return String(v);
}
function computeOutputShape(comp, inputShape, _model) {
  if (!inputShape || !Array.isArray(inputShape)) return null;
  const def = componentRegistry[comp.type];
  if (!def || !def.computeOutputShape) return null;
  try {
    const outputShape = def.computeOutputShape(inputShape, comp.params);
    return Array.isArray(outputShape) ? outputShape : null;
  } catch (e) {
    return null;
  }
}
function getInputShape(comp, model, compMap, connToFrom) {
  if (comp.type === "input") {
    const shape = comp.params.shape;
    return Array.isArray(shape) ? shape : null;
  }
  const srcId = comp.inputs[0] ?? connToFrom.get(comp.id);
  if (!srcId) return null;
  const srcComp = compMap.get(srcId);
  if (!srcComp) return null;
  return getOutputShape(srcComp, model, compMap, connToFrom);
}
function getOutputShape(comp, model, compMap, connToFrom) {
  const inputShape = getInputShape(comp, model, compMap, connToFrom);
  if (!inputShape) return null;
  return computeOutputShape(comp, inputShape, model);
}
var SHARED_EXPERT_MOE_CLASS = `class SharedExpertMoE(nn.Module):
    """Shared-expert MoE: top-k routed experts plus always-on shared experts."""
    def __init__(self, embed_dim, num_experts, num_shared, expert_dim, top_k):
        super().__init__()
        def expert():
            return nn.ModuleDict({
                'gate_proj': nn.Linear(embed_dim, expert_dim, bias=False),
                'up_proj':   nn.Linear(embed_dim, expert_dim, bias=False),
                'down_proj': nn.Linear(expert_dim, embed_dim, bias=False),
            })
        self.router  = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([expert() for _ in range(num_experts)])
        self.shared  = nn.ModuleList([expert() for _ in range(num_shared)])
        self.top_k   = top_k

    @staticmethod
    def _ffn(e, x):
        return e['down_proj'](F.silu(e['gate_proj'](x)) * e['up_proj'](x))

    def forward(self, x):
        scores = self.router(x).softmax(dim=-1)
        top_w, top_i = scores.topk(self.top_k, dim=-1)
        out = x.new_zeros(x.shape) + sum(self._ffn(s, x) for s in self.shared)
        flat_x, flat_o = x.reshape(-1, x.size(-1)), out.reshape(-1, x.size(-1))
        flat_i, flat_w = top_i.reshape(-1, self.top_k), top_w.reshape(-1, self.top_k)
        for e_idx in flat_i.unique():
            hit  = flat_i == e_idx
            rows = hit.any(dim=-1)
            w    = (flat_w * hit).sum(dim=-1)[rows].unsqueeze(-1)
            flat_o[rows] += w * self._ffn(self.experts[int(e_idx)], flat_x[rows])
        return flat_o.view_as(x)

`;
var MOE_FORWARD_METHOD = `
    @staticmethod
    def _moe_forward(moe, x, top_k=2):
        """Top-k routed MoE forward over a {'router', 'experts'} ModuleDict."""
        scores = moe['router'](x).softmax(dim=-1)
        top_w, top_i = scores.topk(top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        flat_x = x.reshape(-1, x.size(-1))
        flat_i = top_i.reshape(-1, top_k)
        flat_w = top_w.reshape(-1, top_k)
        out = torch.zeros_like(flat_x)
        for e_idx in flat_i.unique():
            hit = flat_i == e_idx
            rows = hit.any(dim=-1)
            w = (flat_w * hit).sum(dim=-1)[rows].unsqueeze(-1)
            out[rows] += w * moe['experts'][int(e_idx)](flat_x[rows])
        return out.view_as(x)
`;
var INTENTIONAL_PASSTHROUGH = /* @__PURE__ */ new Set(["positionalEncoding", "rope", "residual", "skipConnection"]);
function isUnsupportedPassthrough(comp) {
  if (comp.type === "input" || comp.type === "output" || comp.type === "stickyNote") return false;
  if (INTENTIONAL_PASSTHROUGH.has(comp.type)) return false;
  return generateForwardCode(comp, "probe", { id: "", name: "", components: [], connections: [], groups: [] }, /* @__PURE__ */ new Map(), /* @__PURE__ */ new Map()) === null;
}
function groupClassName(name) {
  return name.split(/[\s\-_]+/).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join("").replace(/[^a-zA-Z0-9_]/g, "").replace(/^([0-9])/, "_$1") || "Group";
}
function generateGroupSubmodule(group, model, compMap, connToFrom) {
  const memberSet = new Set(group.componentIds);
  const members = model.components.filter((c) => memberSet.has(c.id));
  const className = groupClassName(group.name);
  const memberTopo = topologicalSort(members, {
    components: members,
    connections: model.connections.filter((c) => memberSet.has(c.from) && memberSet.has(c.to)),
    id: "",
    name: "",
    groups: []
  });
  let code = `class ${className}(nn.Module):
`;
  code += `    """${group.name}: auto-generated submodule"""
`;
  code += `    def __init__(self):
`;
  code += `        super().__init__()
`;
  const layerCounter = {};
  const memberLayerNames = {};
  for (const comp of memberTopo) {
    if (comp.type === "input" || comp.type === "output" || comp.type === "stickyNote") continue;
    const baseName = comp.type;
    const count = (layerCounter[baseName] || 0) + 1;
    layerCounter[baseName] = count;
    const layerName = `${baseName}_${count}`;
    memberLayerNames[comp.id] = layerName;
    const layerCode = generateLayerCode(comp, model, compMap, connToFrom);
    if (layerCode) code += `        self.${layerName} = ${layerCode}
`;
  }
  if (memberTopo.some((c) => c.type === "moeLayer")) {
    code += MOE_FORWARD_METHOD;
  }
  code += "\n    def forward(self, x):\n";
  const memberVars = /* @__PURE__ */ new Map();
  for (const comp of memberTopo) {
    if (comp.type === "stickyNote") continue;
    if (comp.type === "input") {
      memberVars.set(comp.id, "x");
      continue;
    }
    if (comp.type === "output") {
      const last = comp.inputs[0] ? memberVars.get(comp.inputs[0]) ?? "x" : "x";
      code += `        return ${last}
`;
      continue;
    }
    const layerName = memberLayerNames[comp.id];
    const fwd = generateForwardCode(comp, layerName, model, memberVars, connToFrom);
    if (fwd) {
      const varName = `${comp.type.replace(/([A-Z])/g, "_$1").toLowerCase().replace(/^_/, "")}_${comp.id.slice(-4)}`;
      memberVars.set(comp.id, varName);
      code += `        ${varName} = ${fwd}
`;
    } else {
      const prevId = comp.inputs[0] ?? connToFrom.get(comp.id);
      const passVar = prevId ? memberVars.get(prevId) ?? "x" : "x";
      memberVars.set(comp.id, passVar);
      if (isUnsupportedPassthrough(comp)) {
        code += `        # TODO: layer '${comp.name}' (${comp.type}) is not yet supported by the exporter; passing through unchanged
`;
      }
    }
  }
  if (!memberTopo.some((c) => c.type === "output")) {
    const lastVar = Array.from(memberVars.values()).pop() ?? "x";
    code += `        return ${lastVar}
`;
  }
  code += "\n";
  return { classCode: code, className };
}
function generatePyTorchCode(model) {
  const components = model.components;
  if (components.length === 0) {
    return "# No components in model";
  }
  const compMap = new Map(components.map((c) => [c.id, c]));
  const connToFrom = new Map(model.connections.map((c) => [c.to, c.from]));
  const eligibleGroups = (model.groups ?? []).filter((g) => {
    const nonIO = g.componentIds.filter((id) => {
      const c = compMap.get(id);
      return c && c.type !== "input" && c.type !== "output";
    });
    return nonIO.length >= 2;
  });
  const topoForGroups = topologicalSort(components, model);
  const groupInfos = [];
  const compToGroupInfo = /* @__PURE__ */ new Map();
  for (const group of eligibleGroups) {
    const memberSet = new Set(group.componentIds);
    const clsName = groupClassName(group.name);
    const attrName = group.name.toLowerCase().replace(/[\s\-]+/g, "_").replace(/[^a-z0-9_]/g, "").replace(/^([0-9])/, "_$1") || "group";
    let entryId = "";
    let exitId = "";
    for (const comp of topoForGroups) {
      if (!memberSet.has(comp.id)) continue;
      if (!entryId) entryId = comp.id;
      exitId = comp.id;
    }
    if (!entryId || !exitId) continue;
    const { classCode } = generateGroupSubmodule(group, model, compMap, connToFrom);
    const info = { group, className: clsName, attrName, memberSet, entryId, exitId, classCode };
    groupInfos.push(info);
    for (const id of group.componentIds) compToGroupInfo.set(id, info);
  }
  const hasAudio = components.some((c) => ["melSpectrogram", "mfcc", "stft", "audioConv"].includes(c.type));
  let code = "# Architecture designed with Neurarch: https://neurarch.com\n";
  code += "# PyTorch: compatible with Python 3.8+ and torch>=1.12\n";
  code += "# Colab: pip install torch torchvision  (usually pre-installed)\n";
  if (hasAudio) code += "# Audio: pip install torchaudio\n";
  const unsupported = components.filter(isUnsupportedPassthrough);
  if (unsupported.length > 0) {
    code += `#
# WARNING: ${unsupported.length} layer(s) below are not yet supported by the PyTorch
`;
    code += "# exporter and pass their input through UNCHANGED in forward():\n";
    for (const c of unsupported) {
      code += `#   - ${c.name} (${c.type})
`;
    }
  }
  code += "\n";
  code += "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom typing import Tuple\n";
  if (hasAudio) code += "import torchaudio\n";
  code += "\n";
  const hp = model.hyperparams ?? {};
  if (Object.keys(hp).length > 0) {
    code += "# \u2500\u2500 Hyperparameters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n";
    for (const [name, def] of Object.entries(hp)) {
      const val = def.type === "str" ? `"${def.value}"` : String(def.value);
      const comment = def.description ? `  # ${def.description}` : "";
      code += `${name} = ${val}${comment}
`;
    }
    code += "\n";
  }
  const customModuleComponents = model.components.filter((c) => c.type === "customModule");
  if (customModuleComponents.length > 0) {
    const emittedIds = /* @__PURE__ */ new Set();
    code += "# \u2500\u2500 Custom Layer Definitions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n";
    for (const comp of customModuleComponents) {
      const layerId = comp.params._customLayerId;
      if (!layerId || emittedIds.has(layerId)) continue;
      emittedIds.add(layerId);
      const def = getCustomLayerById(layerId);
      if (def?.code) code += def.code + "\n\n";
    }
  }
  if (model.components.some((c) => c.type === "sharedExpertMoE")) {
    code += SHARED_EXPERT_MOE_CLASS;
  }
  for (const info of groupInfos) {
    code += info.classCode;
  }
  const className = (model.name || "Model").replace(/\s+/g, "").replace(/-/g, "_").replace(/[^a-zA-Z0-9_]/g, "").replace(/^([0-9])/, "_$1") || "Model";
  code += `class ${className}(nn.Module):
`;
  code += "    def __init__(self):\n";
  code += "        super().__init__()\n\n";
  const layerCounter = {};
  const componentToLayerName = {};
  for (const comp of components) {
    if (comp.type === "input" || comp.type === "output" || comp.type === "stickyNote") continue;
    if (compToGroupInfo.has(comp.id)) continue;
    const baseName = comp.type;
    const count = (layerCounter[baseName] || 0) + 1;
    layerCounter[baseName] = count;
    const layerName = `${baseName}_${count}`;
    componentToLayerName[comp.id] = layerName;
    const layerCode = generateLayerCode(comp, model, compMap, connToFrom);
    if (layerCode) {
      code += `        self.${layerName} = ${layerCode}
`;
      if (comp.augmentations?.includes("freeze")) {
        code += `        self.${layerName}.requires_grad_(False)  # frozen
`;
      }
      if (comp.augmentations?.includes("quantize_int8")) {
        code += `        # TODO: quantize self.${layerName} \u2192 torch.quantization.quantize_dynamic(self.${layerName}, dtype=torch.qint8)
`;
      }
    }
  }
  const addedGroupAttrs = /* @__PURE__ */ new Set();
  for (const info of groupInfos) {
    if (!addedGroupAttrs.has(info.attrName)) {
      code += `        self.${info.attrName} = ${info.className}()  # ${info.group.name}
`;
      addedGroupAttrs.add(info.attrName);
    }
  }
  if (components.some((c) => c.type === "moeLayer" && !compToGroupInfo.has(c.id))) {
    code += MOE_FORWARD_METHOD;
  }
  const inputComponents = components.filter((c) => c.type === "input");
  const hasMultipleInputs = inputComponents.length >= 2;
  if (hasMultipleInputs) {
    code += "\n    def forward(self, src, tgt=None):\n";
  } else {
    code += "\n    def forward(self, x):\n";
  }
  const sortedComponents = topoForGroups;
  const componentVars = /* @__PURE__ */ new Map();
  if (hasMultipleInputs) {
    if (inputComponents[0]) componentVars.set(inputComponents[0].id, "src");
    if (inputComponents[1]) componentVars.set(inputComponents[1].id, "tgt");
  } else {
    const inputComponent = inputComponents[0];
    if (inputComponent) componentVars.set(inputComponent.id, "x");
  }
  let hasReturn = false;
  for (const comp of sortedComponents) {
    if (comp.type === "stickyNote") continue;
    if (comp.type === "input") {
      code += `        # ${comp.name || "Input"} shape: ${JSON.stringify(comp.outputShape || comp.params?.shape || [])}
`;
      continue;
    }
    if (comp.type === "output") {
      const lastVar = componentVars.get(comp.inputs[0] || "") || "x";
      code += `        # Output
`;
      code += `        return ${lastVar}
`;
      hasReturn = true;
      continue;
    }
    const groupInfo = compToGroupInfo.get(comp.id);
    if (groupInfo) {
      if (comp.id === groupInfo.entryId) {
        const prevId = comp.inputs[0] ?? connToFrom.get(comp.id);
        const inputVar = prevId ? componentVars.get(prevId) ?? "x" : "x";
        const varName = `${groupInfo.attrName}_out`;
        code += `        ${varName} = self.${groupInfo.attrName}(${inputVar})  # ${groupInfo.group.name}
`;
        for (const mid of groupInfo.group.componentIds) componentVars.set(mid, varName);
      }
      continue;
    }
    const layerName = componentToLayerName[comp.id];
    const forwardCode = generateForwardCode(comp, layerName, model, componentVars, connToFrom);
    if (forwardCode) {
      const baseName = comp.type.replace(/([A-Z])/g, "_$1").toLowerCase().replace(/^_/, "");
      const idSuffix = comp.id.slice(-6).replace(/[^a-z0-9]/gi, "_");
      const varName = `${baseName}_${idSuffix}`;
      componentVars.set(comp.id, varName);
      if (comp.augmentations?.includes("gradient_checkpoint")) {
        code += `        ${varName} = torch.utils.checkpoint.checkpoint(self.${layerName}, ${forwardCode.match(/\(([^)]+)\)/)?.[1] ?? "x"})  # gradient checkpoint
`;
      } else if (comp.augmentations?.includes("amp")) {
        code += `        with torch.autocast(device_type='cuda'):  # amp
`;
        code += `            ${varName} = ${forwardCode}
`;
      } else {
        code += `        ${varName} = ${forwardCode}
`;
      }
    } else {
      const prevId = comp.inputs[0] ?? connToFrom.get(comp.id);
      const passthroughVar = prevId ? componentVars.get(prevId) ?? "x" : "x";
      componentVars.set(comp.id, passthroughVar);
      if (comp.type === "positionalEncoding") {
        const prevComp = prevId ? compMap.get(prevId) : null;
        if (prevComp?.type === "conv2d") {
          code += `        ${passthroughVar} = ${passthroughVar}.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
`;
          code += `        # positionalEncoding: add learned or sinusoidal PE here
`;
        } else {
          code += `        # positionalEncoding: add positional encoding externally (e.g. sinusoidal or learned PE)
`;
        }
      } else if (comp.type === "rope") {
        code += `        # rope: RoPE applied inside attention (no separate layer needed)
`;
      } else if (comp.type === "residual" || comp.type === "skipConnection") {
        code += `        # Residual connection: wire shortcut manually if needed
`;
      } else if (isUnsupportedPassthrough(comp)) {
        code += `        # TODO: layer '${comp.name}' (${comp.type}) is not yet supported by the exporter; passing through unchanged
`;
      }
    }
  }
  if (!hasReturn) {
    const outputComponent = sortedComponents.find((c) => c.type === "output");
    if (outputComponent && outputComponent.inputs.length > 0) {
      const outputVar = componentVars.get(outputComponent.inputs[0]) || "x";
      code += `        return ${outputVar}
`;
    } else {
      const lastVar = Array.from(componentVars.values()).pop() || "x";
      code += `        return ${lastVar}
`;
    }
  }
  const hasEmbedding = sortedComponents.some((c) => c.type === "embedding");
  const hasConv2d = sortedComponents.some((c) => c.type === "conv2d");
  const makeExampleShape = (comp) => {
    const raw = comp?.params?.shape ?? comp?.outputShape;
    if (raw && raw.length > 0) return raw[0] === 1 ? raw : [1, ...raw];
    if (hasEmbedding) return [1, 128];
    if (!hasConv2d) return [1, 64];
    return [1, 3, 224, 224];
  };
  code += `

if __name__ == '__main__':
`;
  code += `    model = ${className}()
`;
  code += `    model.eval()

`;
  if (hasMultipleInputs) {
    const srcShape = makeExampleShape(inputComponents[0]);
    const tgtShape = makeExampleShape(inputComponents[1]);
    code += `    src = torch.randint(0, ${inputComponents[0].params?.shape?.[0] ?? 37e3}, (${srcShape.join(", ")}))  # (batch, src_seq_len)
`;
    code += `    tgt = torch.randint(0, ${inputComponents[1].params?.shape?.[0] ?? 37e3}, (${tgtShape.join(", ")}))  # (batch, tgt_seq_len)
`;
    code += `    with torch.no_grad():
`;
    code += `        output = model(src, tgt)

`;
    code += `    print(f'Src shape    : {tuple(src.shape)}')
`;
    code += `    print(f'Tgt shape    : {tuple(tgt.shape)}')
`;
    code += `    print(f'Output shape : {tuple(output.shape)}')
`;
  } else {
    const inputComp = inputComponents[0];
    const exampleShape = makeExampleShape(inputComp);
    const shapeStr = exampleShape.join(", ");
    let dimLabels;
    if (exampleShape.length === 2) {
      dimLabels = ["batch", "features"];
    } else if (exampleShape.length === 3) {
      dimLabels = exampleShape[1] < 64 ? ["batch", "seq_len", "embed_dim"] : ["batch", "channels", "length"];
    } else if (exampleShape.length === 4) {
      dimLabels = ["batch", "channels", "height", "width"];
    } else {
      dimLabels = exampleShape.map((_, i) => `dim_${i}`);
    }
    const shapeComment = `# (${dimLabels.join(", ")})`;
    const firstNonIO = sortedComponents.find((c) => c.type !== "input" && c.type !== "output");
    const isTokenModel = firstNonIO?.type === "embedding" || exampleShape.length <= 2;
    const vocabSize = firstNonIO?.type === "embedding" ? firstNonIO.params?.vocabSize || 5e4 : 5e4;
    if (isTokenModel) {
      code += `    x = torch.randint(0, ${vocabSize}, (${shapeStr}))  ${shapeComment}
`;
    } else {
      code += `    x = torch.randn(${shapeStr})  ${shapeComment}
`;
    }
    code += `    with torch.no_grad():
`;
    code += `        output = model(x)

`;
    code += `    print(f'Input  shape : {tuple(x.shape)}')
`;
    code += `    print(f'Output shape : {tuple(output.shape)}')
`;
  }
  code += `    total = sum(p.numel() for p in model.parameters())
`;
  code += `    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
`;
  code += `    print(f'Parameters   : {total:,} total, {trainable:,} trainable')
`;
  return code;
}
function resolveSequenceInputSize(comp, inputShape, compMap) {
  if (inputShape && Array.isArray(inputShape)) {
    if (inputShape.length === 3) return inputShape[2];
    if (inputShape.length === 2) return inputShape[1];
  }
  const visited = /* @__PURE__ */ new Set();
  const findEmbed = (id) => {
    if (visited.has(id)) return null;
    visited.add(id);
    const c = compMap.get(id);
    if (!c) return null;
    if (c.type === "embedding") return c.params.embedDim || c.params.embeddingDim || 128;
    for (const inId of c.inputs) {
      const r = findEmbed(inId);
      if (r !== null) return r;
    }
    return null;
  };
  for (const inId of comp.inputs) {
    const embedDim = findEmbed(inId);
    if (embedDim !== null) return embedDim;
  }
  return "input_size";
}
function resolveParam(value, hp) {
  if (typeof value === "string" && value.startsWith("$")) {
    const varName = value.slice(1);
    return varName in hp ? varName : value;
  }
  return value;
}
function generateLayerCode(comp, model, compMap, connToFrom) {
  const { type } = comp;
  const hp = model.hyperparams ?? {};
  const params = Object.fromEntries(
    Object.entries(comp.params).map(([k, v]) => [k, resolveParam(v, hp)])
  );
  const inputShape = getInputShape(comp, model, compMap, connToFrom);
  switch (type) {
    case "linear": {
      let inFeatures = "in_features";
      if (inputShape && Array.isArray(inputShape)) {
        if (inputShape.length === 1) {
          inFeatures = inputShape[0];
        } else {
          const upstreamComp = comp.inputs[0] ? compMap.get(comp.inputs[0]) : null;
          const upstreamType = upstreamComp?.type || "";
          if ([
            "lstm",
            "gru",
            "rnn",
            "bidirectionalLSTM",
            "multiHeadAttention",
            "transformerBlock",
            "feedForward",
            "layerNorm"
          ].includes(upstreamType)) {
            inFeatures = inputShape[inputShape.length - 1];
          } else {
            inFeatures = inputShape.reduce((a, b) => a * b, 1);
          }
        }
      }
      if (inFeatures === "in_features") {
        const PASSTHROUGH = /* @__PURE__ */ new Set([
          "dropout",
          "relu",
          "gelu",
          "sigmoid",
          "tanh",
          "swish",
          "leakyRelu",
          "softmax",
          "layerNorm",
          "positionalEncoding",
          "rope"
        ]);
        let upId = comp.inputs[0];
        while (upId) {
          const upComp = compMap.get(upId);
          if (!upComp) break;
          if (["lstm", "gru", "rnn", "bidirectionalLSTM"].includes(upComp.type)) {
            const h = upComp.params.hiddenSize || 128;
            const mult = upComp.type === "bidirectionalLSTM" || upComp.params.bidirectional ? 2 : 1;
            inFeatures = h * mult;
            break;
          }
          if (!PASSTHROUGH.has(upComp.type)) break;
          upId = upComp.inputs[0];
        }
      }
      if (params.inFeatures !== void 0) {
        inFeatures = params.inFeatures;
      }
      return `nn.Linear(${inFeatures}, ${params.outFeatures})`;
    }
    case "conv2d":
      let inChannels = "in_channels";
      if (inputShape && Array.isArray(inputShape)) {
        if (inputShape.length === 4) {
          inChannels = inputShape[1];
        } else if (inputShape.length === 3) {
          inChannels = inputShape[0];
        } else if (inputShape.length === 2) {
          inChannels = inputShape[0];
        }
      }
      if (params.inChannels !== void 0) {
        inChannels = params.inChannels;
      }
      return `nn.Conv2d(${inChannels}, ${params.outChannels ?? 32}, kernel_size=${pyDim(params.kernelSize, 3)}, stride=${pyDim(params.stride, 1)}, padding=${pyDim(params.padding, 0)})`;
    case "conv1d": {
      let inChannels1d = "in_channels";
      if (inputShape && Array.isArray(inputShape) && inputShape.length >= 2) {
        inChannels1d = inputShape[inputShape.length - 2];
      }
      if (params.inChannels !== void 0) {
        inChannels1d = params.inChannels;
      }
      return `nn.Conv1d(${inChannels1d}, ${params.outChannels ?? 32}, kernel_size=${pyDim(params.kernelSize, 3)}, stride=${pyDim(params.stride, 1)}, padding=${pyDim(params.padding, 0)})`;
    }
    case "conv3d": {
      const c3dIn = params.inChannels ?? (inputShape?.[1] ?? "in_channels");
      return `nn.Conv3d(${c3dIn}, ${params.outChannels || 32}, kernel_size=${pyDim(params.kernelSize, 3)}, stride=${pyDim(params.stride, 1)}, padding=${pyDim(params.padding, 0)})`;
    }
    case "depthwiseConv2d": {
      const dwIn = params.inChannels ?? (inputShape?.[0] ?? "in_channels");
      const dm = params.depthMultiplier || 1;
      return `nn.Conv2d(${dwIn}, ${dwIn}*${dm}, kernel_size=${pyDim(params.kernelSize, 3)}, groups=${dwIn}, bias=True)`;
    }
    case "separableConv2d": {
      const scIn = params.inChannels ?? (inputShape?.[0] ?? "in_channels");
      const scOut = params.outChannels || scIn;
      const scK = pyDim(params.kernelSize, 3);
      return `nn.Sequential(
            nn.Conv2d(${scIn}, ${scIn}, kernel_size=${scK}, groups=${scIn}, bias=False),
            nn.Conv2d(${scIn}, ${scOut}, kernel_size=1)
        )`;
    }
    case "transposeConv2d": {
      const tcIn = params.inChannels ?? (inputShape?.[0] ?? "in_channels");
      return `nn.ConvTranspose2d(${tcIn}, ${params.outChannels || 32}, kernel_size=${pyDim(params.kernelSize, 2)}, stride=${pyDim(params.stride, 2)}, padding=${pyDim(params.padding, 0)})`;
    }
    case "maxpool2d":
      return `nn.MaxPool2d(kernel_size=${pyDim(params.kernelSize, 2)}, stride=${pyDim(params.stride || params.kernelSize, 2)}, padding=${pyDim(params.padding, 0)})`;
    case "avgpool2d":
      return `nn.AvgPool2d(kernel_size=${pyDim(params.kernelSize, 2)}, stride=${pyDim(params.stride || params.kernelSize, 2)}, padding=${pyDim(params.padding, 0)})`;
    case "dropout":
      return `nn.Dropout(p=${params.p || 0.5})`;
    case "batchNorm": {
      let numFeatures = "num_features";
      let bnClass = "BatchNorm2d";
      if (inputShape && Array.isArray(inputShape)) {
        if (inputShape.length === 3) {
          numFeatures = inputShape[0];
          bnClass = "BatchNorm2d";
        } else if (inputShape.length === 4) {
          numFeatures = inputShape[1];
          bnClass = "BatchNorm2d";
        } else if (inputShape.length === 2) {
          numFeatures = inputShape[0];
          bnClass = "BatchNorm1d";
        } else if (inputShape.length === 1) {
          numFeatures = inputShape[0];
          bnClass = "BatchNorm1d";
        }
      } else if (params.numFeatures !== void 0) {
        numFeatures = params.numFeatures;
        bnClass = "BatchNorm2d";
      }
      return `nn.${bnClass}(${numFeatures})`;
    }
    case "flatten":
      return null;
    // Flatten is functional
    case "embedding":
      return `nn.Embedding(${params.numEmbeddings || params.vocabSize || 1e4}, ${params.embeddingDim || params.embedDim || 128})`;
    case "embeddingBag":
      return `nn.EmbeddingBag(${params.numEmbeddings || 1e4}, ${params.embeddingDim || 64}, mode='mean')`;
    case "layerNorm":
      let normFeatures = "normalized_shape";
      if (inputShape && Array.isArray(inputShape) && inputShape.length > 0) {
        normFeatures = inputShape[inputShape.length - 1];
      } else if (params.normalizedShape) {
        normFeatures = Array.isArray(params.normalizedShape) ? params.normalizedShape[0] : params.normalizedShape;
      }
      return `nn.LayerNorm(${normFeatures})`;
    case "multiHeadAttention":
      const embedDim = params.hiddenDim || params.embedDim || 512;
      const numHeads = params.numHeads || 8;
      return `nn.MultiheadAttention(embed_dim=${embedDim}, num_heads=${numHeads}, batch_first=True)`;
    case "feedForward":
      const hiddenDim = params.hiddenDim || params.embedDim || 512;
      const ffDim = params.ffDim || 2048;
      return `nn.Sequential(
            nn.Linear(${hiddenDim}, ${ffDim}),
            nn.ReLU(),
            nn.Linear(${ffDim}, ${hiddenDim})
        )`;
    case "lstm": {
      let lstmInputSize = resolveSequenceInputSize(comp, inputShape, compMap);
      if (params.inputSize !== void 0) lstmInputSize = params.inputSize;
      const lstmLayers = params.numLayers || 1;
      const lstmBidir = params.bidirectional ? ", bidirectional=True" : "";
      return `nn.LSTM(${lstmInputSize}, ${params.hiddenSize || 128}, num_layers=${lstmLayers}, batch_first=True${lstmBidir})`;
    }
    case "gru": {
      let gruInputSize = resolveSequenceInputSize(comp, inputShape, compMap);
      if (params.inputSize !== void 0) gruInputSize = params.inputSize;
      const gruLayers = params.numLayers || 1;
      const gruBidir = params.bidirectional ? ", bidirectional=True" : "";
      return `nn.GRU(${gruInputSize}, ${params.hiddenSize || 128}, num_layers=${gruLayers}, batch_first=True${gruBidir})`;
    }
    case "rnn": {
      let rnnInputSize = resolveSequenceInputSize(comp, inputShape, compMap);
      if (params.inputSize !== void 0) rnnInputSize = params.inputSize;
      return `nn.RNN(${rnnInputSize}, ${params.hiddenSize || 128}, num_layers=${params.numLayers || 1}, batch_first=True)`;
    }
    case "bidirectionalLSTM": {
      let biLstmInput = resolveSequenceInputSize(comp, inputShape, compMap);
      if (params.inputSize !== void 0) biLstmInput = params.inputSize;
      return `nn.LSTM(${biLstmInput}, ${params.hiddenSize || 128}, num_layers=${params.numLayers || 1}, batch_first=True, bidirectional=True)`;
    }
    case "gelu":
      return `nn.GELU()`;
    case "swish":
      return `nn.SiLU()`;
    case "leakyRelu":
      return `nn.LeakyReLU(negative_slope=${params.negativeSlope || 0.01})`;
    case "elu":
      return `nn.ELU(alpha=${params.alpha || 1})`;
    case "prelu":
      return `nn.PReLU(num_parameters=${params.numParameters || 1})`;
    case "softmax":
      return `nn.Softmax(dim=-1)`;
    case "globalAvgPool2d":
      return `nn.AdaptiveAvgPool2d(1)`;
    case "adaptiveAvgPool2d":
    case "adaptiveMaxPool2d": {
      const rawOut = params.outputSize ?? 1;
      const outStr = Array.isArray(rawOut) ? `(${rawOut.join(", ")})` : rawOut;
      const cls = type === "adaptiveMaxPool2d" ? "AdaptiveMaxPool2d" : "AdaptiveAvgPool2d";
      return `nn.${cls}(${outStr})`;
    }
    case "windowAttention": {
      const waDim = params.embedDim || 96;
      const waHeads = params.numHeads || 3;
      return `nn.MultiheadAttention(embed_dim=${waDim}, num_heads=${waHeads}, batch_first=True)  # Window attention (simplified)`;
    }
    case "localAttention": {
      const laDim = params.embedDim || 512;
      const laHeads = params.numHeads || 8;
      const laWin = params.windowSize || 256;
      return `nn.MultiheadAttention(embed_dim=${laDim}, num_heads=${laHeads}, batch_first=True)  # Sliding window ${laWin} (simplified as full MHA; add a band mask for the real window)`;
    }
    case "linearAttention": {
      const linDim = params.embedDim || 512;
      const linHeads = params.numHeads || 8;
      return `nn.MultiheadAttention(embed_dim=${linDim}, num_heads=${linHeads}, batch_first=True)  # Linear attention (simplified as MHA; swap in a kernel-feature map for O(n))`;
    }
    case "learnedPositionalEmbedding": {
      const lpMax = params.maxLen || 512;
      const lpDim = params.embedDim || 768;
      return `nn.Embedding(${lpMax}, ${lpDim})  # learned absolute positional embedding`;
    }
    case "mamba": {
      const mmD = params.dModel || 256;
      const mmState = params.dState || 16;
      const mmConv = params.dConv || 4;
      const mmExp = params.expand || 2;
      return `nn.Identity()  # Mamba(d_model=${mmD}, d_state=${mmState}, d_conv=${mmConv}, expand=${mmExp}), pip install mamba-ssm and swap in`;
    }
    case "causalAttention": {
      const caDim = params.embedDim || 512;
      const caHeads = params.numHeads || 8;
      return `nn.MultiheadAttention(embed_dim=${caDim}, num_heads=${caHeads}, batch_first=True)`;
    }
    case "groupedQueryAttention": {
      const D = params.embedDim || 4096;
      const H = params.numHeads || 32;
      const Hkv = params.numKVHeads || 8;
      const hD = Math.floor(D / H);
      return `nn.ModuleDict({
            'q_proj': nn.Linear(${D}, ${D},        bias=False),   # ${H} heads \xD7 ${hD}
            'k_proj': nn.Linear(${D}, ${Hkv * hD}, bias=False),   # ${Hkv} KV heads \xD7 ${hD}
            'v_proj': nn.Linear(${D}, ${Hkv * hD}, bias=False),
            'o_proj': nn.Linear(${D}, ${D},        bias=False),
        })  # GQA: ${H}Q / ${Hkv}KV heads (requires F.scaled_dot_product_attention)`;
    }
    case "mla": {
      const D = params.embedDim || 4096;
      const H = params.numHeads || 32;
      const kvL = params.kvLatentDim || 512;
      const hD = Math.floor(D / H);
      return `nn.ModuleDict({
            'q_proj':  nn.Linear(${D}, ${D},   bias=False),   # ${H} heads \xD7 ${hD}
            'kv_down': nn.Linear(${D}, ${kvL}, bias=False),   # compress K/V into the cached latent
            'k_up':    nn.Linear(${kvL}, ${D}, bias=False),   # reconstruct per-head K from the latent
            'v_up':    nn.Linear(${kvL}, ${D}, bias=False),   # reconstruct per-head V from the latent
            'o_proj':  nn.Linear(${D}, ${D},   bias=False),
        })  # MLA: at inference, cache the ${kvL}-dim latent instead of per-head K/V (simplified: decoupled-RoPE key path omitted)`;
    }
    case "sharedExpertMoE": {
      const D = params.embedDim || 4096;
      const E = params.numExperts || 64;
      const S = params.numSharedExperts || 2;
      const I = params.expertDim || 1408;
      const K = params.topK || 6;
      return `SharedExpertMoE(embed_dim=${D}, num_experts=${E}, num_shared=${S}, expert_dim=${I}, top_k=${K})`;
    }
    case "maxpool1d":
      return `nn.MaxPool1d(kernel_size=${pyDim(params.kernelSize, 2)}, stride=${pyDim(params.stride || params.kernelSize, 2)})`;
    case "avgpool1d":
      return `nn.AvgPool1d(kernel_size=${pyDim(params.kernelSize, 2)}, stride=${pyDim(params.stride || params.kernelSize, 2)})`;
    case "pixelShuffle":
      return `nn.PixelShuffle(${params.upscaleFactor || 2})`;
    case "rmsNorm": {
      const rmsShape = params.normalizedShape || inputShape?.[inputShape.length - 1] || 512;
      return `nn.RMSNorm(${rmsShape})`;
    }
    case "swiglu": {
      const D = params.embedDim ?? params.inFeatures ?? 4096;
      const I = params.intermediateSize ?? params.hiddenFeatures ?? params.ffDim ?? Math.round(D * 8 / 3);
      return `nn.ModuleDict({
            'gate_proj': nn.Linear(${D}, ${I}, bias=False),
            'up_proj':   nn.Linear(${D}, ${I}, bias=False),
            'down_proj': nn.Linear(${I}, ${D}, bias=False),
        })  # SwiGLU FFN (LLaMA-style)`;
    }
    case "moeLayer": {
      const D = params.embedDim ?? 512;
      const E = params.numExperts ?? 8;
      const K = params.topK ?? 2;
      const I = params.expertDim ?? params.ffDim ?? Math.round(D * 8 / 3);
      return `nn.ModuleDict({
            'router': nn.Linear(${D}, ${E}, bias=False),
            'experts': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(${D}, ${I}, bias=False), nn.SiLU(),
                    nn.Linear(${I}, ${D}, bias=False),
                ) for _ in range(${E})
            ]),
        })  # MoE top-${K}`;
    }
    case "patchEmbed": {
      const P = params.patchSize ?? 16;
      const D = params.embedDim ?? 768;
      const inC = params.inChans ?? params.inChannels ?? 3;
      return `nn.Conv2d(${inC}, ${D}, kernel_size=${P}, stride=${P})  # Patch embedding (ViT-style)`;
    }
    case "seBlock": {
      const C = params.channels ?? (inputShape?.[0] ?? 64);
      const r = params.reductionRatio ?? params.reduction ?? 16;
      const mid = Math.max(1, Math.floor(C / r));
      return `nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(${C}, ${mid}), nn.ReLU(),
            nn.Linear(${mid}, ${C}), nn.Sigmoid(),
        )  # SE Block`;
    }
    case "layerScale":
      return null;
    // nn.Parameter, not a module — handled in forward pass
    case "alibi":
    case "dropPath":
      return null;
    // Applied externally / stochastic depth via timm
    case "reshape":
      return null;
    // handled in forward pass
    case "upsample":
      return `nn.Upsample(scale_factor=${params.scaleFactor || 2}, mode='${params.mode || "nearest"}')`;
    case "instanceNorm": {
      let inChannels2 = "num_features";
      let inClass = "InstanceNorm2d";
      if (inputShape && Array.isArray(inputShape)) {
        if (inputShape.length === 4) {
          inChannels2 = inputShape[1];
        } else if (inputShape.length === 3) {
          inChannels2 = inputShape[0];
        } else if (inputShape.length === 2) {
          inChannels2 = inputShape[0];
          inClass = "InstanceNorm1d";
        }
      }
      if (params.numFeatures !== void 0) inChannels2 = params.numFeatures;
      return `nn.${inClass}(${inChannels2})`;
    }
    case "groupNorm": {
      let gnChannels = "num_channels";
      if (inputShape && Array.isArray(inputShape)) {
        if (inputShape.length === 4) gnChannels = inputShape[1];
        else if (inputShape.length === 3 || inputShape.length === 2) gnChannels = inputShape[0];
      }
      if (params.numChannels !== void 0) gnChannels = params.numChannels;
      return `nn.GroupNorm(${params.numGroups || 32}, ${gnChannels})`;
    }
    case "transformerBlock": {
      const tbDim = params.hiddenDim || params.embedDim || 512;
      const tbHeads = params.numHeads || 8;
      const tbFf = params.ffDim || tbDim * 4;
      const isDecoder = comp.inputs.length >= 2;
      if (isDecoder) {
        return `nn.TransformerDecoderLayer(d_model=${tbDim}, nhead=${tbHeads}, dim_feedforward=${tbFf}, batch_first=True)`;
      }
      return `nn.TransformerEncoderLayer(d_model=${tbDim}, nhead=${tbHeads}, dim_feedforward=${tbFf}, batch_first=True)`;
    }
    case "selfAttention":
    case "attention": {
      const attnDim = params.embedDim || 128;
      const attnHeads = params.numHeads || 8;
      return `nn.MultiheadAttention(embed_dim=${attnDim}, num_heads=${attnHeads}, batch_first=True)`;
    }
    case "crossAttention": {
      const xaDim = params.embedDim || 512;
      const xaHeads = params.numHeads || 8;
      return `nn.MultiheadAttention(embed_dim=${xaDim}, num_heads=${xaHeads}, batch_first=True)`;
    }
    case "crossModalAttention": {
      const cmaDim = params.embedDim || 256;
      const cmaHeads = params.numHeads || 8;
      return `nn.MultiheadAttention(embed_dim=${cmaDim}, num_heads=${cmaHeads}, batch_first=True)`;
    }
    case "positionalEncoding":
    case "residual":
    case "skipConnection":
    case "rope":
      return null;
    // Architecture patterns — not standard nn.Module layers
    case "add":
    case "multiply":
    case "concatenate":
      return null;
    // These are functional operations
    // ── RL heads ─────────────────────────────────────────────────────────────────
    case "dqnHead": {
      const dqnH = params.hiddenSize ?? (inputShape?.[inputShape.length - 1] ?? 512);
      const dqnA = params.numActions || 18;
      return `nn.Linear(${dqnH}, ${dqnA})`;
    }
    case "actorHead": {
      const actH = inputShape?.[inputShape.length - 1] ?? 256;
      return `nn.Linear(${actH}, ${params.numActions || 6})`;
    }
    case "criticHead": {
      const crtH = inputShape?.[inputShape.length - 1] ?? 256;
      return `nn.Linear(${crtH}, ${params.outputDim || 1})`;
    }
    case "policyNetwork": {
      const polIn = inputShape?.[inputShape.length - 1] ?? "obs_dim";
      const polH = params.hiddenSize || 256;
      return `nn.Sequential(
            nn.Linear(${polIn}, ${polH}),
            nn.ReLU(),
            nn.Linear(${polH}, ${params.numActions || 6})
        )`;
    }
    case "valueNetwork": {
      const valIn = inputShape?.[inputShape.length - 1] ?? "obs_dim";
      const valH = params.hiddenSize || 256;
      return `nn.Sequential(
            nn.Linear(${valIn}, ${valH}),
            nn.ReLU(),
            nn.Linear(${valH}, 1)
        )`;
    }
    // ── Graph ─────────────────────────────────────────────────────────────────────
    case "graphConv":
    case "gcn":
      return null;
    // Requires torch_geometric: GCNConv — not standard nn.Module
    case "graphAttention":
    case "gat":
      return null;
    // Requires torch_geometric: GATConv
    case "graphSAGE":
      return null;
    // Requires torch_geometric: SAGEConv
    // ── Tabular ───────────────────────────────────────────────────────────────────
    case "tabnet": {
      const tnIn = params.inputDim ?? (inputShape?.[inputShape.length - 1] ?? "input_dim");
      const tnOut = params.outputDim || 64;
      return `nn.Sequential(
            nn.Linear(${tnIn}, ${tnOut}),
            nn.GELU()
        )  # Simplified: use pytorch-tabnet for full TabNet`;
    }
    case "featureInteraction":
      return null;
    // Custom FM/DCN interaction — no standard nn.Module
    // ── Multimodal ────────────────────────────────────────────────────────────────
    case "fusion": {
      const fusIn = params.fusionDim ?? (inputShape?.[inputShape.length - 1] ?? "in_dim");
      const fusOut = params.outputDim || 256;
      return `nn.Linear(${fusIn}, ${fusOut})`;
    }
    case "projection": {
      const prjIn = params.inDim ?? (inputShape?.[inputShape.length - 1] ?? "in_dim");
      return `nn.Linear(${prjIn}, ${params.outDim || 512})`;
    }
    // ── Audio ────────────────────────────────────────────────────────────────────
    case "audioConv": {
      const acInC = params.inChannels || 1;
      const acOutC = params.outChannels || 32;
      const acK = pyDim(params.kernelSize, 3);
      return `nn.Conv1d(${acInC}, ${acOutC}, kernel_size=${acK}, stride=${params.stride || 1}, padding=${params.padding || 0})`;
    }
    case "melSpectrogram":
      return null;
    // torchaudio.transforms.MelSpectrogram — not an nn.Module inline
    case "mfcc":
      return null;
    // torchaudio.transforms.MFCC
    case "stft":
      return null;
    // torch.stft — functional
    // Element-wise activations emitted functionally in the forward pass
    // (F.relu / torch.sigmoid / torch.tanh) — there is no nn.Module to declare
    // in __init__, so returning null here is correct, not a gap. Listed
    // explicitly so they don't trip the "No layer code" warning below.
    case "relu":
    case "sigmoid":
    case "tanh":
      return null;
    default:
      console.warn(`[codeGenerator] No layer code for component type: "${comp.type}". Generated code may be incomplete.`);
      return null;
  }
}
function generateForwardCode(comp, layerName, _model, componentVars, connToFrom) {
  const { type, inputs } = comp;
  const getInputVars = () => {
    if (inputs.length === 0) {
      const fromId = connToFrom.get(comp.id);
      if (fromId) return [componentVars.get(fromId) ?? "x"];
      return ["x"];
    }
    return inputs.map((inputId) => componentVars.get(inputId) || "x").filter(Boolean);
  };
  const inputVars = getInputVars();
  const primaryVar = inputVars[0] || "x";
  switch (type) {
    case "linear":
    case "conv2d":
    case "conv1d":
    case "conv3d":
    case "depthwiseConv2d":
    case "separableConv2d":
    case "transposeConv2d":
    case "maxpool2d":
    case "avgpool2d":
    case "maxpool1d":
    case "avgpool1d":
    case "dropout":
    case "batchNorm":
    case "pixelShuffle":
    case "rmsNorm":
      return `self.${layerName}(${primaryVar})`;
    case "relu":
      return `F.relu(${primaryVar})`;
    case "elu":
      return `F.elu(${primaryVar})`;
    case "prelu":
      return `self.${layerName}(${primaryVar})`;
    case "sigmoid":
      return `torch.sigmoid(${primaryVar})`;
    case "tanh":
      return `torch.tanh(${primaryVar})`;
    case "flatten":
      return `torch.flatten(${primaryVar}, 1)`;
    case "reshape": {
      const rshape = comp.params.shape || [512];
      const shapeStr = Array.isArray(rshape) ? rshape.join(", ") : rshape;
      return `${primaryVar}.reshape(${primaryVar}.size(0), ${shapeStr})`;
    }
    case "embedding":
    case "embeddingBag":
    case "layerNorm":
    case "gelu":
    case "swish":
    case "leakyRelu":
    case "softmax":
    case "upsample":
    case "instanceNorm":
    case "groupNorm":
      return `self.${layerName}(${primaryVar})`;
    case "globalAvgPool2d":
      return `self.${layerName}(${primaryVar}).flatten(1)`;
    case "adaptiveAvgPool2d":
    case "adaptiveMaxPool2d": {
      const outSize = comp.params?.outputSize ?? 1;
      const needsFlatten = outSize === 1 || Array.isArray(outSize) && outSize.every((s) => s === 1);
      return `self.${layerName}(${primaryVar})${needsFlatten ? ".flatten(1)" : ""}`;
    }
    case "windowAttention":
    case "causalAttention":
    case "localAttention":
    case "linearAttention":
      return `self.${layerName}(${primaryVar}, ${primaryVar}, ${primaryVar})[0]`;
    case "mamba":
      return `self.${layerName}(${primaryVar})`;
    case "learnedPositionalEmbedding":
      return `${primaryVar} + self.${layerName}(torch.arange(${primaryVar}.size(1), device=${primaryVar}.device))`;
    case "groupedQueryAttention": {
      const gH = comp.params.numHeads || 32;
      const gHkv = comp.params.numKVHeads || 8;
      const gD = comp.params.embedDim || 4096;
      const ghD = Math.floor(gD / gH);
      const rep = Math.floor(gH / gHkv);
      const lm = `self.${layerName}`;
      return `${lm}['o_proj'](F.scaled_dot_product_attention(
            ${lm}['q_proj'](${primaryVar}).view(${primaryVar}.size(0),-1,${gH},${ghD}).transpose(1,2),
            ${lm}['k_proj'](${primaryVar}).view(${primaryVar}.size(0),-1,${gHkv},${ghD}).transpose(1,2).repeat_interleave(${rep},dim=1),
            ${lm}['v_proj'](${primaryVar}).view(${primaryVar}.size(0),-1,${gHkv},${ghD}).transpose(1,2).repeat_interleave(${rep},dim=1),
            is_causal=True,
        ).transpose(1,2).reshape(${primaryVar}.size(0),-1,${gD}))`;
    }
    case "mla": {
      const mD = comp.params.embedDim || 4096;
      const mH = comp.params.numHeads || 32;
      const mhD = Math.floor(mD / mH);
      const lm = `self.${layerName}`;
      return `${lm}['o_proj'](F.scaled_dot_product_attention(
            ${lm}['q_proj'](${primaryVar}).view(${primaryVar}.size(0),-1,${mH},${mhD}).transpose(1,2),
            ${lm}['k_up'](${lm}['kv_down'](${primaryVar})).view(${primaryVar}.size(0),-1,${mH},${mhD}).transpose(1,2),
            ${lm}['v_up'](${lm}['kv_down'](${primaryVar})).view(${primaryVar}.size(0),-1,${mH},${mhD}).transpose(1,2),
            is_causal=True,
        ).transpose(1,2).reshape(${primaryVar}.size(0),-1,${mD}))  # cache kv_down(x), not K/V`;
    }
    case "sharedExpertMoE":
      return `self.${layerName}(${primaryVar})`;
    case "swiglu":
      return `self.${layerName}['down_proj'](F.silu(self.${layerName}['gate_proj'](${primaryVar})) * self.${layerName}['up_proj'](${primaryVar}))`;
    case "patchEmbed":
      return `self.${layerName}(${primaryVar}).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]`;
    case "seBlock":
      return `${primaryVar} * self.${layerName}(${primaryVar}).view(${primaryVar}.size(0), -1, 1, 1)`;
    case "moeLayer":
      return `self._moe_forward(self.${layerName}, ${primaryVar}, top_k=${comp.params?.topK ?? 2})`;
    case "alibi":
    case "dropPath":
    case "layerScale":
      return `${primaryVar}  # ${comp.type} applied externally`;
    case "lstm":
    case "gru":
    case "rnn":
    case "bidirectionalLSTM":
      return comp.params?.returnSequences === false ? `self.${layerName}(${primaryVar})[0][:, -1, :]` : `self.${layerName}(${primaryVar})[0]`;
    case "multiHeadAttention":
      return `self.${layerName}(${primaryVar}, ${primaryVar}, ${primaryVar})[0]`;
    case "transformerBlock":
      if (inputVars.length >= 2) {
        return `self.${layerName}(${inputVars[0]}, ${inputVars[1]})`;
      }
      return `self.${layerName}(${primaryVar})`;
    case "selfAttention":
    case "attention":
    case "crossModalAttention":
      return `self.${layerName}(${primaryVar}, ${primaryVar}, ${primaryVar})[0]`;
    case "crossAttention": {
      const kvVar = inputVars.length >= 2 ? inputVars[1] : primaryVar;
      return `self.${layerName}(${primaryVar}, ${kvVar}, ${kvVar})[0]`;
    }
    case "feedForward":
      return `self.${layerName}(${primaryVar})`;
    case "add":
      if (inputVars.length === 1) {
        return `${primaryVar}`;
      } else if (inputVars.length === 2) {
        return `${inputVars[0]} + ${inputVars[1]}`;
      } else {
        return inputVars.join(" + ");
      }
    case "multiply":
      if (inputVars.length === 1) {
        return `${primaryVar}`;
      } else if (inputVars.length === 2) {
        return `${inputVars[0]} * ${inputVars[1]}`;
      } else {
        return inputVars.join(" * ");
      }
    case "concatenate":
      if (inputVars.length === 1) {
        return `${primaryVar}`;
      } else {
        const dim = comp.params?.dim !== void 0 ? comp.params.dim : -1;
        return `torch.cat([${inputVars.join(", ")}], dim=${dim})`;
      }
    case "dqnHead":
    case "actorHead":
    case "criticHead":
    case "policyNetwork":
    case "valueNetwork":
    case "fusion":
    case "projection":
    case "tabnet":
    case "audioConv":
      return `self.${layerName}(${primaryVar})`;
    case "graphConv":
    case "gcn":
    case "graphAttention":
    case "gat":
    case "graphSAGE":
      return `self.${layerName}(${primaryVar}, edge_index)  # pass edge_index from graph data`;
    case "featureInteraction":
      return `${primaryVar}  # feature interaction: implement FM/DCN manually`;
    case "melSpectrogram":
      return `torchaudio.transforms.MelSpectrogram(sample_rate=${comp.params.sampleRate || 16e3}, n_mels=${comp.params.nMels || 80})(${primaryVar})`;
    case "mfcc":
      return `torchaudio.transforms.MFCC(sample_rate=${comp.params.sampleRate || 16e3}, n_mfcc=${comp.params.nMfcc || 40})(${primaryVar})`;
    case "stft":
      return `torch.stft(${primaryVar}, n_fft=${comp.params.nFft || 512}, return_complex=True).abs()`;
    default:
      return null;
  }
}
function topologicalSort(components, _model) {
  const componentMap = new Map(components.map((c) => [c.id, c]));
  const visited = /* @__PURE__ */ new Set();
  const result = [];
  function visit(id) {
    if (visited.has(id)) return;
    visited.add(id);
    const comp = componentMap.get(id);
    if (!comp) return;
    for (const inputId of comp.inputs) {
      visit(inputId);
    }
    result.push(comp);
  }
  const inputComponents = components.filter((c) => c.type === "input" || c.inputs.length === 0);
  for (const input of inputComponents) {
    visit(input.id);
  }
  for (const comp of components) {
    if (!visited.has(comp.id)) {
      visit(comp.id);
    }
  }
  return result;
}

// ../Neurarch/node_modules/zustand/esm/vanilla.mjs
var createStoreImpl = (createState) => {
  let state;
  const listeners = /* @__PURE__ */ new Set();
  const setState = (partial, replace) => {
    const nextState = typeof partial === "function" ? partial(state) : partial;
    if (!Object.is(nextState, state)) {
      const previousState = state;
      state = (replace != null ? replace : typeof nextState !== "object" || nextState === null) ? nextState : Object.assign({}, state, nextState);
      listeners.forEach((listener) => listener(state, previousState));
    }
  };
  const getState = () => state;
  const getInitialState = () => initialState;
  const subscribe = (listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };
  const destroy = () => {
    if ((import.meta.env ? import.meta.env.MODE : void 0) !== "production") {
      console.warn(
        "[DEPRECATED] The `destroy` method will be unsupported in a future version. Instead use unsubscribe function returned by subscribe. Everything will be garbage-collected if store is garbage-collected."
      );
    }
    listeners.clear();
  };
  const api = { setState, getState, getInitialState, subscribe, destroy };
  const initialState = state = createState(setState, getState, api);
  return api;
};
var createStore = (createState) => createState ? createStoreImpl(createState) : createStoreImpl;

// ../Neurarch/node_modules/zustand/esm/index.mjs
var import_react = __toESM(require_react(), 1);
var import_with_selector = __toESM(require_with_selector(), 1);
var { useDebugValue } = import_react.default;
var { useSyncExternalStoreWithSelector } = import_with_selector.default;
var didWarnAboutEqualityFn = false;
var identity = (arg) => arg;
function useStore(api, selector = identity, equalityFn) {
  if ((import.meta.env ? import.meta.env.MODE : void 0) !== "production" && equalityFn && !didWarnAboutEqualityFn) {
    console.warn(
      "[DEPRECATED] Use `createWithEqualityFn` instead of `create` or use `useStoreWithEqualityFn` instead of `useStore`. They can be imported from 'zustand/traditional'. https://github.com/pmndrs/zustand/discussions/1937"
    );
    didWarnAboutEqualityFn = true;
  }
  const slice = useSyncExternalStoreWithSelector(
    api.subscribe,
    api.getState,
    api.getServerState || api.getInitialState,
    selector,
    equalityFn
  );
  useDebugValue(slice);
  return slice;
}
var createImpl = (createState) => {
  if ((import.meta.env ? import.meta.env.MODE : void 0) !== "production" && typeof createState !== "function") {
    console.warn(
      "[DEPRECATED] Passing a vanilla store will be unsupported in a future version. Instead use `import { useStore } from 'zustand'`."
    );
  }
  const api = typeof createState === "function" ? createStore(createState) : createState;
  const useBoundStore = (selector, equalityFn) => useStore(api, selector, equalityFn);
  Object.assign(useBoundStore, api);
  return useBoundStore;
};
var create = (createState) => createState ? createImpl(createState) : createImpl;

// src/utils/providerStore.ts
var PROVIDER_DEFS = {
  gemini: {
    id: "gemini",
    label: "Gemini",
    icon: "\u2726",
    keyPlaceholder: "AIza...",
    keyLabel: "Google AI Studio key",
    keyHint: "Get free key at aistudio.google.com",
    defaultBaseUrl: "https://generativelanguage.googleapis.com",
    openaiCompat: false,
    // 2.5-flash is the default. We had 2.0-flash before (15 RPM vs 5 RPM
    // looked attractive), but Google removed 2.0-flash from the free tier
    // for many accounts in 2025 — it returns 429 with `limit: 0` for those
    // users and BYOK can never recover. 2.5-flash is the conservative
    // choice: actually free for everyone today.
    defaultModel: "gemini-2.5-flash",
    models: [
      { id: "gemini-2.5-flash", label: "2.5 Flash (recommended)", contextK: 1e3, tier: "fast" },
      { id: "gemini-2.5-flash-lite", label: "2.5 Flash Lite", contextK: 1e3, tier: "fast" },
      { id: "gemini-2.0-flash", label: "2.0 Flash (paid only)", contextK: 1e3, tier: "pro" },
      { id: "gemini-1.5-flash", label: "1.5 Flash", contextK: 1e3, tier: "free" }
    ]
  },
  claude: {
    id: "claude",
    label: "Claude",
    icon: "\u25C6",
    keyPlaceholder: "sk-ant-api...",
    keyLabel: "Anthropic API key",
    keyHint: "Get key at console.anthropic.com",
    defaultBaseUrl: "https://api.anthropic.com",
    openaiCompat: false,
    defaultModel: "claude-opus-4-8",
    models: [
      { id: "claude-opus-4-8", label: "Opus 4.8", contextK: 1e3, tier: "ultra" },
      { id: "claude-opus-4-6", label: "Opus 4.6", contextK: 1e3, tier: "ultra" },
      { id: "claude-sonnet-4-6", label: "Sonnet 4.6", contextK: 1e3, tier: "pro" },
      { id: "claude-haiku-4-5-20251001", label: "Haiku 4.5", contextK: 200, tier: "fast" }
    ]
  },
  openai: {
    id: "openai",
    label: "OpenAI",
    icon: "\u2B21",
    keyPlaceholder: "sk-...",
    keyLabel: "OpenAI API key",
    keyHint: "Get key at platform.openai.com",
    defaultBaseUrl: "https://api.openai.com/v1",
    openaiCompat: true,
    defaultModel: "gpt-4o",
    models: [
      { id: "gpt-4o", label: "GPT-4o", contextK: 128, tier: "pro" },
      { id: "gpt-4o-mini", label: "GPT-4o Mini", contextK: 128, tier: "fast" },
      { id: "o3-mini", label: "o3-mini", contextK: 200, tier: "ultra" }
    ]
  },
  groq: {
    id: "groq",
    label: "Groq",
    icon: "\u26A1",
    keyPlaceholder: "gsk_...",
    keyLabel: "Groq API key",
    keyHint: "Free key at console.groq.com: very fast inference",
    defaultBaseUrl: "https://api.groq.com/openai/v1",
    openaiCompat: true,
    // Catalog refreshed against api.groq.com/openai/v1/models 2026-08-21: the
    // whole 2024 lineup (llama-3.3/3.1, mixtral, gemma2) has been retired and
    // returns model_not_found; every call with the old default failed.
    defaultModel: "openai/gpt-oss-120b",
    models: [
      { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B", contextK: 128, tier: "pro" },
      { id: "openai/gpt-oss-20b", label: "GPT-OSS 20B", contextK: 128, tier: "fast" },
      { id: "qwen/qwen3.6-27b", label: "Qwen 3.6 27B", contextK: 128, tier: "fast" }
    ]
  },
  // OpenRouter — one key unlocks a fleet of free + cheap open models. The
  // backend proxy serves these from our shared OPENROUTER_API_KEY pool (no
  // per-user key needed when signed in); BYOK users can paste their own.
  // `:free` models share a global rate limit, so they're best as the free
  // tier / fallback. For a true never-throttle experience under load, fund a
  // small OpenRouter balance and switch the default to the paid DeepSeek line.
  openrouter: {
    id: "openrouter",
    label: "OpenRouter",
    icon: "\u{1F6E4}\uFE0F",
    keyPlaceholder: "sk-or-...",
    keyLabel: "OpenRouter API key",
    keyHint: "Free key at openrouter.ai/keys, one key, many free open models",
    defaultBaseUrl: "https://openrouter.ai/api/v1",
    openaiCompat: true,
    // DeepSeek V3 is the reliable default (real provider capacity, no per-minute
    // throttling like the ":free" models). Proxy overrides the served model by
    // plan anyway; this default matters for BYOK users + the dropdown.
    defaultModel: "deepseek/deepseek-chat",
    // IDs verified live on openrouter.ai/api/v1/models (2026-06-02). NOTE:
    // DeepSeek / Qwen-2.5 have no ":free" variant on OpenRouter — only list
    // free ids that actually exist or requests 404. (The proxy overrides the
    // served model by plan anyway; this list is mostly for BYOK users.)
    models: [
      { id: "meta-llama/llama-3.3-70b-instruct:free", label: "Llama 3.3 70B (free)", contextK: 128, tier: "free" },
      { id: "qwen/qwen3-next-80b-a3b-instruct:free", label: "Qwen3 Next 80B (free)", contextK: 32, tier: "free" },
      { id: "meta-llama/llama-3.2-3b-instruct:free", label: "Llama 3.2 3B (free, fast)", contextK: 128, tier: "free" },
      { id: "deepseek/deepseek-chat", label: "DeepSeek V3 (cheap, paid)", contextK: 64, tier: "pro" }
    ]
  },
  // xAI / Grok — OpenAI-compatible chat completions at api.x.ai. BYOK-only
  // (like OpenAI / Ollama): the user pastes their own xAI key and the browser
  // talks to api.x.ai directly, so there is no server key or proxy quota to
  // manage. embedDim-style routing aside, Grok speaks the same /v1/chat shape
  // as Groq, so the generic openai-compat dispatch handles it with no special
  // case in agentClient.
  xai: {
    id: "xai",
    label: "Grok (xAI)",
    icon: "\u2715",
    keyPlaceholder: "xai-...",
    keyLabel: "xAI API key",
    keyHint: "Get key at console.x.ai: BYOK, talks to api.x.ai directly",
    defaultBaseUrl: "https://api.x.ai/v1",
    openaiCompat: true,
    defaultModel: "grok-4",
    models: [
      { id: "grok-4", label: "Grok 4", contextK: 256, tier: "ultra" },
      { id: "grok-3", label: "Grok 3", contextK: 131, tier: "pro" },
      { id: "grok-3-mini", label: "Grok 3 Mini", contextK: 131, tier: "fast" }
    ]
  },
  ollama: {
    id: "ollama",
    label: "Ollama",
    icon: "\u{1F999}",
    keyPlaceholder: "ollama (no key needed)",
    keyLabel: "Local model: no key required",
    keyHint: "Runs locally at localhost:11434, install at ollama.com",
    defaultBaseUrl: "http://localhost:11434/v1",
    openaiCompat: true,
    defaultModel: "llama3.2",
    models: [
      { id: "llama3.2", label: "LLaMA 3.2", contextK: 128, tier: "pro" },
      { id: "llama3.1", label: "LLaMA 3.1", contextK: 128, tier: "pro" },
      { id: "mistral", label: "Mistral 7B", contextK: 32, tier: "fast" },
      { id: "qwen2.5-coder", label: "Qwen 2.5 Coder", contextK: 128, tier: "pro" },
      { id: "deepseek-r1", label: "DeepSeek R1", contextK: 64, tier: "ultra" }
    ]
  },
  // Custom / self-hosted — any server that speaks the OpenAI /v1/chat/completions
  // shape: LM Studio, vLLM, LiteLLM, LocalAI, text-generation-webui, an internal
  // gateway, etc. The whole point is on-prem: teams that cannot send a proprietary
  // architecture to a third-party cloud API point the agent at their own endpoint.
  // Base URL and model are free-text; the key is optional (blank for keyless local
  // servers, sent as a Bearer token otherwise). BYOK-direct, never proxied.
  custom: {
    id: "custom",
    label: "Custom / Local",
    icon: "\u{1F50C}",
    keyPlaceholder: "optional: blank for keyless local servers",
    keyLabel: "API key (optional)",
    keyHint: "Any OpenAI-compatible server: LM Studio (:1234), vLLM (:8000), LiteLLM, LocalAI. Set the base URL and model below.",
    defaultBaseUrl: "http://localhost:1234/v1",
    openaiCompat: true,
    defaultModel: "",
    models: []
  }
};
var LS = {
  get: (k) => {
    try {
      return localStorage.getItem(k) ?? "";
    } catch {
      return "";
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {
    }
  }
};
function migrate(k) {
  const ls = LS.get(k);
  if (ls) return ls;
  try {
    const ss = sessionStorage.getItem(k);
    if (ss) {
      LS.set(k, ss);
      sessionStorage.removeItem(k);
      return ss;
    }
  } catch {
  }
  return "";
}
function keyForProvider(id) {
  return migrate(`llm-key-${id}`);
}
function setKeyForProvider(id, key) {
  LS.set(`llm-key-${id}`, key);
}
function modelForProvider(id) {
  const stored = migrate(`llm-model-${id}`);
  if (id === "gemini" && stored === "gemini-2.0-flash") {
    LS.set(`llm-model-${id}`, PROVIDER_DEFS[id].defaultModel);
    return PROVIDER_DEFS[id].defaultModel;
  }
  return stored || PROVIDER_DEFS[id].defaultModel;
}
function setModelForProvider(id, model) {
  LS.set(`llm-model-${id}`, model);
}
function baseUrlForProvider(id) {
  return migrate(`llm-baseurl-${id}`) || PROVIDER_DEFS[id].defaultBaseUrl;
}
function setBaseUrlForProvider(id, url) {
  LS.set(`llm-baseurl-${id}`, url);
}
var HF_TOKEN_KEY = "hf-token";
function hfToken() {
  return migrate(HF_TOKEN_KEY);
}
function setHFTokenLS(t) {
  LS.set(HF_TOKEN_KEY, t);
}
var KAGGLE_USERNAME_KEY = "kaggle-username";
var KAGGLE_KEY_KEY = "kaggle-key";
function kaggleCreds() {
  return { username: LS.get(KAGGLE_USERNAME_KEY) ?? "", key: LS.get(KAGGLE_KEY_KEY) ?? "" };
}
function setKaggleCredsLS(username, key) {
  LS.set(KAGGLE_USERNAME_KEY, username.trim());
  LS.set(KAGGLE_KEY_KEY, key.trim());
}
var AUTO_TIER_KEY = "llm-auto-tier";
var AUTO_TIER_SEEN_KEY = "llm-auto-tier-seen";
var HAS_PRIOR_LLM_USAGE_KEY = "llm-provider";
function autoTierEnabled() {
  const v = migrate(AUTO_TIER_KEY);
  if (v === "1") return true;
  if (v === "0") return false;
  const isReturning = !!migrate(HAS_PRIOR_LLM_USAGE_KEY);
  const defaultOn = !isReturning;
  LS.set(AUTO_TIER_KEY, defaultOn ? "1" : "0");
  LS.set(AUTO_TIER_SEEN_KEY, "1");
  return defaultOn;
}
function setAutoTier(on) {
  LS.set(AUTO_TIER_KEY, on ? "1" : "0");
  LS.set(AUTO_TIER_SEEN_KEY, "1");
}
var PLAN_VERIFIER_KEY = "agent-plan-verifier";
function planVerifierEnabled() {
  return LS.get(PLAN_VERIFIER_KEY) === "1";
}
function setPlanVerifier(on) {
  LS.set(PLAN_VERIFIER_KEY, on ? "1" : "0");
}
var ARCH_COHERENCE_VERIFIER_KEY = "arch-coherence-verifier";
function archCoherenceVerifierEnabled() {
  return LS.get(ARCH_COHERENCE_VERIFIER_KEY) === "1";
}
function setArchCoherenceVerifier(on) {
  LS.set(ARCH_COHERENCE_VERIFIER_KEY, on ? "1" : "0");
}
var CONVERGENCE_LOOP_KEY = "agent-convergence-loop";
function convergenceLoopEnabled() {
  return LS.get(CONVERGENCE_LOOP_KEY) === "1";
}
function setConvergenceLoop(on) {
  LS.set(CONVERGENCE_LOOP_KEY, on ? "1" : "0");
}
var CONVERGENCE_TARGET_KEY = "agent-convergence-target";
function convergenceTargetGrade() {
  const v = LS.get(CONVERGENCE_TARGET_KEY);
  return v === "B" || v === "C" ? v : "A";
}
function setConvergenceTargetGrade(g) {
  LS.set(CONVERGENCE_TARGET_KEY, g);
}
var STRUCTURAL_REPAIR_KEY = "agent-structural-repair";
function structuralRepairEnabled() {
  return LS.get(STRUCTURAL_REPAIR_KEY) !== "0";
}
function setStructuralRepair(on) {
  LS.set(STRUCTURAL_REPAIR_KEY, on ? "1" : "0");
}
var GROUNDING_RETRIEVAL_KEY = "agent-grounding-retrieval";
function groundingRetrievalEnabled() {
  return LS.get(GROUNDING_RETRIEVAL_KEY) !== "0";
}
function setGroundingRetrieval(on) {
  LS.set(GROUNDING_RETRIEVAL_KEY, on ? "1" : "0");
}
var PARAM_EXPLOSION_KEY = "agent-param-explosion-threshold";
function paramExplosionThreshold() {
  const v = migrate(PARAM_EXPLOSION_KEY);
  const n = v ? parseFloat(v) : NaN;
  return Number.isFinite(n) && n >= 2 && n <= 50 ? n : 5;
}
function setParamExplosionThreshold(n) {
  const clamped = Math.max(2, Math.min(50, n));
  LS.set(PARAM_EXPLOSION_KEY, String(clamped));
}
var COLOR_BY_TYPE_KEY = "canvas-color-by-type";
function colorByTypeEnabled() {
  return LS.get(COLOR_BY_TYPE_KEY) === "1";
}
function setColorByType(on) {
  LS.set(COLOR_BY_TYPE_KEY, on ? "1" : "0");
}
var NODE_DENSITY_COMPACT_KEY = "canvas-node-density-compact";
function nodeDensityCompact() {
  return LS.get(NODE_DENSITY_COMPACT_KEY) === "1";
}
function setNodeDensityCompact(on) {
  LS.set(NODE_DENSITY_COMPACT_KEY, on ? "1" : "0");
}
var LOD_KEY = "canvas-lod";
function lodEnabled() {
  const v = LS.get(LOD_KEY);
  return !v ? true : v === "1";
}
function setLod(on) {
  LS.set(LOD_KEY, on ? "1" : "0");
}
var PAPER_MODE_KEY = "canvas-paper-mode";
function paperModeEnabled() {
  const v = LS.get(PAPER_MODE_KEY);
  return !v ? true : v === "1";
}
function setPaperMode(on) {
  LS.set(PAPER_MODE_KEY, on ? "1" : "0");
}
var FIGURE_ATTRIBUTION_KEY = "figure-attribution";
function figureAttributionEnabled() {
  const v = LS.get(FIGURE_ATTRIBUTION_KEY);
  return !v ? true : v === "1";
}
function setFigureAttribution(on) {
  LS.set(FIGURE_ATTRIBUTION_KEY, on ? "1" : "0");
}
var SHAPE_AWARE_KEY = "canvas-shape-aware-sizing";
function shapeAwareSizingEnabled() {
  return LS.get(SHAPE_AWARE_KEY) === "1";
}
function setShapeAwareSizing(on) {
  LS.set(SHAPE_AWARE_KEY, on ? "1" : "0");
}
var AUTO_FOLD_KEY = "canvas-auto-fold-repeats";
function autoFoldEnabled() {
  return LS.get(AUTO_FOLD_KEY) === "1";
}
function setAutoFold(on) {
  LS.set(AUTO_FOLD_KEY, on ? "1" : "0");
}
var PROVIDER_OR_MIGRATION_KEY = "llm-provider-or-migrated";
function resolveInitialProvider() {
  const stored = migrate("llm-provider");
  try {
    if (!LS.get(PROVIDER_OR_MIGRATION_KEY)) {
      LS.set(PROVIDER_OR_MIGRATION_KEY, "1");
      if (stored === "gemini" && !migrate("llm-key-gemini")) {
        LS.set("llm-provider", "openrouter");
        return "openrouter";
      }
    }
  } catch {
  }
  return stored || "openrouter";
}
var useProviderStore = create((set, get) => ({
  // Default to OpenRouter for new users — free open models served from our
  // shared OPENROUTER_API_KEY pool give the most generous "keep asking" free
  // experience (15/day vs Gemini's 3/day), with key rotation + model fallback
  // for resilience under load. Gemini/Claude stay available in the picker.
  // Returning users keep their explicit choice; see resolveInitialProvider for
  // the one-time Gemini→OpenRouter nudge.
  activeProvider: resolveInitialProvider(),
  setActiveProvider: (id) => {
    LS.set("llm-provider", id);
    set({ activeProvider: id });
  },
  getKey: (id) => keyForProvider(id ?? get().activeProvider),
  setKey: (id, key) => {
    setKeyForProvider(id, key);
    set({});
  },
  getModel: (id) => modelForProvider(id ?? get().activeProvider),
  setModel: (id, model) => {
    setModelForProvider(id, model);
    set({});
  },
  getBaseUrl: (id) => baseUrlForProvider(id ?? get().activeProvider),
  setBaseUrl: (id, url) => {
    setBaseUrlForProvider(id, url);
    set({});
  },
  getHFToken: () => hfToken(),
  setHFToken: (token) => {
    setHFTokenLS(token.trim());
    set({});
  },
  getKaggleCreds: () => kaggleCreds(),
  setKaggleCreds: (username, key) => {
    setKaggleCredsLS(username, key);
    set({});
  },
  activeKey: () => keyForProvider(get().activeProvider),
  activeModel: () => modelForProvider(get().activeProvider),
  activeDef: () => PROVIDER_DEFS[get().activeProvider],
  autoTierEnabled: () => autoTierEnabled(),
  setAutoTier: (on) => {
    setAutoTier(on);
    set({});
  },
  paramExplosionThreshold: () => paramExplosionThreshold(),
  setParamExplosionThreshold: (n) => {
    setParamExplosionThreshold(n);
    set({});
  },
  planVerifierEnabled: () => planVerifierEnabled(),
  setPlanVerifier: (on) => {
    setPlanVerifier(on);
    set({});
  },
  archCoherenceVerifierEnabled: () => archCoherenceVerifierEnabled(),
  setArchCoherenceVerifier: (on) => {
    setArchCoherenceVerifier(on);
    set({});
  },
  structuralRepairEnabled: () => structuralRepairEnabled(),
  setStructuralRepair: (on) => {
    setStructuralRepair(on);
    set({});
  },
  convergenceLoopEnabled: () => convergenceLoopEnabled(),
  setConvergenceLoop: (on) => {
    setConvergenceLoop(on);
    set({});
  },
  convergenceTargetGrade: () => convergenceTargetGrade(),
  setConvergenceTargetGrade: (g) => {
    setConvergenceTargetGrade(g);
    set({});
  },
  groundingRetrievalEnabled: () => groundingRetrievalEnabled(),
  setGroundingRetrieval: (on) => {
    setGroundingRetrieval(on);
    set({});
  },
  colorByTypeEnabled: () => colorByTypeEnabled(),
  setColorByType: (on) => {
    setColorByType(on);
    set({});
  },
  nodeDensityCompact: () => nodeDensityCompact(),
  setNodeDensityCompact: (on) => {
    setNodeDensityCompact(on);
    set({});
  },
  lodEnabled: () => lodEnabled(),
  setLod: (on) => {
    setLod(on);
    set({});
  },
  paperModeEnabled: () => paperModeEnabled(),
  setPaperMode: (on) => {
    setPaperMode(on);
    set({});
  },
  figureAttributionEnabled: () => figureAttributionEnabled(),
  setFigureAttribution: (on) => {
    setFigureAttribution(on);
    set({});
  },
  shapeAwareSizingEnabled: () => shapeAwareSizingEnabled(),
  setShapeAwareSizing: (on) => {
    setShapeAwareSizing(on);
    set({});
  },
  autoFoldEnabled: () => autoFoldEnabled(),
  setAutoFold: (on) => {
    setAutoFold(on);
    set({});
  }
}));

// src/utils/hfModelLoader.ts
function hfAuthHeaders() {
  if (typeof process !== "undefined" && process.env?.HF_TOKEN) {
    return { Authorization: `Bearer ${process.env.HF_TOKEN.trim()}` };
  }
  try {
    const token = useProviderStore.getState().getHFToken().trim();
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}
async function fetchHFRealParamCount(modelId, headers) {
  try {
    const r = await fetch(`https://huggingface.co/api/models/${modelId}`, { headers });
    if (!r.ok) return void 0;
    const info = await r.json();
    const total = info?.safetensors?.total;
    return typeof total === "number" && total > 0 ? total : void 0;
  } catch {
    return void 0;
  }
}
var ARCH_DEFAULTS = {
  // LLM families
  mistral: { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, intermediate_size: 14336, vocab_size: 32e3, max_position_embeddings: 32768 },
  mixtral: { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, intermediate_size: 14336, vocab_size: 32e3, max_position_embeddings: 32768 },
  falcon: { hidden_size: 4544, num_hidden_layers: 32, num_attention_heads: 71, intermediate_size: 18176, vocab_size: 65024, max_position_embeddings: 2048 },
  phi: { hidden_size: 2048, num_hidden_layers: 24, num_attention_heads: 32, intermediate_size: 8192, vocab_size: 51200, max_position_embeddings: 2048 },
  phi3: { hidden_size: 3072, num_hidden_layers: 32, num_attention_heads: 32, intermediate_size: 8192, vocab_size: 32064, max_position_embeddings: 4096 },
  qwen2: { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, intermediate_size: 11008, vocab_size: 151936, max_position_embeddings: 32768 },
  gemma: { hidden_size: 3072, num_hidden_layers: 28, num_attention_heads: 16, intermediate_size: 24576, vocab_size: 256e3, max_position_embeddings: 8192 },
  gemma2: { hidden_size: 3584, num_hidden_layers: 46, num_attention_heads: 16, intermediate_size: 14336, vocab_size: 256e3, max_position_embeddings: 8192 },
  deepseek: { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, intermediate_size: 11008, vocab_size: 102400, max_position_embeddings: 4096 },
  cohere: { hidden_size: 8192, num_hidden_layers: 40, num_attention_heads: 64, intermediate_size: 22528, vocab_size: 256e3, max_position_embeddings: 8192 },
  // Audio / seq2seq
  whisper: {
    hidden_size: 512,
    num_hidden_layers: 6,
    num_attention_heads: 8,
    intermediate_size: 2048,
    vocab_size: 51865,
    d_model: 512,
    encoder_layers: 6,
    decoder_layers: 6,
    encoder_attention_heads: 8,
    decoder_attention_heads: 8,
    encoder_ffn_dim: 2048,
    decoder_ffn_dim: 2048
  },
  wav2vec2: { hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12, intermediate_size: 3072, vocab_size: 32 },
  // CV families
  swin: { hidden_size: 96, num_hidden_layers: 12, num_attention_heads: 3 },
  convnext: { hidden_size: 96, num_hidden_layers: 12, num_attention_heads: 1 },
  deit: { hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12 },
  beit: { hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12 }
};
function normalizeAndAugmentConfig(config, modelId) {
  const mIdLower = modelId.toLowerCase();
  const className = config._class_name;
  if (className && typeof className === "string") {
    if (!config.model_type) config.model_type = className.toLowerCase();
    if (!config.architectures || config.architectures.length === 0) config.architectures = [className];
    if (/unet|transformer2dmodel|transformer3dmodel|dit|denois|vae|autoencoder/i.test(className)) {
      config._diffusion = true;
    }
  }
  {
    const NESTED_TEXT_KEYS = [
      "text_config",
      "llm_config",
      "language_config",
      "language_model_config",
      "decoder_config"
    ];
    for (const k of NESTED_TEXT_KEYS) {
      const sub = config[k];
      if (sub && typeof sub === "object" && (sub.hidden_size || sub.d_model || sub.n_embd || sub.num_hidden_layers || sub.num_layers)) {
        for (const [sk, sv] of Object.entries(sub)) {
          if (config[sk] === void 0 || config[sk] === null) {
            config[sk] = sv;
          }
        }
        if (sub.model_type) config._textModelType = sub.model_type;
        break;
      }
    }
  }
  if (!config.hidden_size) {
    if (config.d_model) config.hidden_size = config.d_model;
    else if (config.n_embd) config.hidden_size = config.n_embd;
    else if (config.model_dim) config.hidden_size = config.model_dim;
  }
  if (!config.num_hidden_layers) {
    if (config.n_layer) config.num_hidden_layers = config.n_layer;
    else if (config.num_layers) config.num_hidden_layers = config.num_layers;
    else if (config.encoder_layers) config.num_hidden_layers = config.encoder_layers;
    else if (config.num_blocks) config.num_hidden_layers = config.num_blocks;
  }
  if (!config.num_attention_heads) {
    if (config.n_head) config.num_attention_heads = config.n_head;
    else if (config.encoder_attention_heads) config.num_attention_heads = config.encoder_attention_heads;
    else if (config.num_heads) config.num_attention_heads = config.num_heads;
  }
  if (!config.intermediate_size) {
    if (config.n_inner) config.intermediate_size = config.n_inner;
    else if (config.encoder_ffn_dim) config.intermediate_size = config.encoder_ffn_dim;
    else if (config.d_ff) config.intermediate_size = config.d_ff;
    else if (config.ffn_dim) config.intermediate_size = config.ffn_dim;
  }
  if (!config.hidden_size) {
    const mTypeLower = (config.model_type || "").toLowerCase();
    for (const [key, defs] of Object.entries(ARCH_DEFAULTS)) {
      if (mTypeLower.includes(key) || mIdLower.includes(key)) {
        for (const [k, v] of Object.entries(defs)) {
          if (config[k] === void 0 || config[k] === null) {
            config[k] = v;
          }
        }
        if (!config.hidden_size && config.d_model) config.hidden_size = config.d_model;
        break;
      }
    }
  }
  return config;
}
async function fetchDiffusersConfig(modelId) {
  try {
    const miResp = await fetch(`https://huggingface.co/${modelId}/resolve/main/model_index.json`);
    if (!miResp.ok) return null;
    const mi = await miResp.json();
    if (!mi.transformer && !mi.unet) return null;
    const denoiser = mi.transformer ? "transformer" : "unet";
    const subResp = await fetch(`https://huggingface.co/${modelId}/resolve/main/${denoiser}/config.json`);
    if (!subResp.ok) return null;
    const raw = await subResp.json();
    const num = (...keys) => {
      for (const k of keys) if (typeof raw[k] === "number") return raw[k];
      return void 0;
    };
    const className = raw._class_name || (Array.isArray(mi[denoiser]) ? mi[denoiser][1] : void 0) || (denoiser === "unet" ? "UNet2DConditionModel" : "Transformer2DModel");
    const heads = num("num_attention_heads");
    const headDim = num("attention_head_dim");
    const hidden = (heads && headDim ? heads * headDim : void 0) ?? num("inner_dim", "cross_attention_dim", "hidden_size", "caption_channels", "joint_attention_dim") ?? 1152;
    const layers = denoiser === "transformer" ? num("num_layers", "num_hidden_layers", "depth") ?? 12 : void 0;
    const config = normalizeAndAugmentConfig({
      model_type: denoiser === "unet" ? "unet" : "dit",
      architectures: [className],
      hidden_size: hidden,
      num_hidden_layers: layers,
      num_attention_heads: heads,
      // Use the model's real FFN width when it ships one (Wan: ffn_dim 8960,
      // not hidden×4 = 6144 — the hardcoded ×4 undercounted the MLP by ~260M
      // across 30 layers). Falls back to ×4 only when no width is declared.
      intermediate_size: num("ffn_dim", "intermediate_size", "d_ff", "mlp_dim", "mlp_ratio") ?? hidden * 4
    }, modelId);
    const c = config;
    c._diffusion = true;
    c._denoiser = denoiser;
    c._diffuser_class = className;
    c.in_channels = num("in_channels") ?? 4;
    c.sample_size = num("sample_size") ?? 64;
    c.patch_size = num("patch_size") ?? 2;
    c.cross_attention_dim = num("cross_attention_dim", "caption_channels", "joint_attention_dim", "text_dim", "text_embed_dim");
    if (Array.isArray(raw.block_out_channels)) c.block_out_channels = raw.block_out_channels;
    if (typeof raw.layers_per_block === "number") c.layers_per_block = raw.layers_per_block;
    const classLower = className.toLowerCase();
    if (/sd3|stablediffusion3|sd35|\bflux\b|mmdit/.test(classLower) || typeof raw.joint_attention_dim === "number") {
      c._mmdit = true;
    }
    const singleLayers = num("num_single_layers");
    if (singleLayers != null) c._num_single_layers = singleLayers;
    if (mi.vae) {
      try {
        const vr = await fetch(`https://huggingface.co/${modelId}/resolve/main/vae/config.json`);
        if (vr.ok) {
          const vraw = await vr.json();
          c._vae = {
            in_channels: typeof vraw.in_channels === "number" ? vraw.in_channels : 3,
            latent_channels: typeof vraw.latent_channels === "number" ? vraw.latent_channels : c.in_channels ?? 4,
            block_out_channels: Array.isArray(vraw.block_out_channels) ? vraw.block_out_channels : [128, 256, 512, 512],
            layers_per_block: typeof vraw.layers_per_block === "number" ? vraw.layers_per_block : 2
          };
        }
      } catch {
      }
    }
    return config;
  } catch {
    return null;
  }
}
async function fetchHFModelConfig(modelId) {
  try {
    const authHeaders = hfAuthHeaders();
    const configResponse = await fetch(
      `https://huggingface.co/${modelId}/resolve/main/config.json`,
      { headers: authHeaders }
    );
    if (configResponse.ok) {
      const raw = await configResponse.json();
      const config2 = normalizeAndAugmentConfig(raw, modelId);
      config2._configSource = "config";
      config2._realParamCount = await fetchHFRealParamCount(modelId, authHeaders);
      return config2;
    }
    const diffusion = await fetchDiffusersConfig(modelId);
    if (diffusion) {
      diffusion._configSource = diffusion._configSource ?? "diffusers";
      return diffusion;
    }
    const apiResponse = await fetch(`https://huggingface.co/api/models/${modelId}`, { headers: authHeaders });
    const modelInfo = apiResponse.ok ? await apiResponse.json() : {};
    const modelIdLower = modelId.toLowerCase();
    const pipelineTag = modelInfo.pipeline_tag ?? "";
    let inferredType = modelInfo.config?.model_type ?? "";
    if (!inferredType || !ARCH_DEFAULTS[inferredType]) {
      for (const key of Object.keys(ARCH_DEFAULTS)) {
        if ((inferredType || pipelineTag || modelIdLower).toLowerCase().includes(key) || modelIdLower.includes(key)) {
          inferredType = key;
          break;
        }
      }
    }
    const defaults = ARCH_DEFAULTS[inferredType] ?? {};
    const config = normalizeAndAugmentConfig({
      model_type: inferredType || pipelineTag || "unknown",
      architectures: modelInfo.config?.architectures ?? [],
      vocab_size: modelInfo.config?.vocab_size,
      ...defaults
    }, modelId);
    if (!config.hidden_size) {
      throw new Error(`Cannot determine architecture for "${modelId}", config.json not accessible and no known defaults`);
    }
    config._configSource = "fallback";
    const realTotal = modelInfo?.safetensors?.total;
    if (typeof realTotal === "number" && realTotal > 0) {
      config._realParamCount = realTotal;
    }
    return config;
  } catch (error) {
    console.error(`Error fetching HF model config for ${modelId}:`, error);
    return null;
  }
}
function buildDiffusionDenoiser(a) {
  const { config, components, connections, generateId, generateConnId, xPos, verticalSpacing: vs } = a;
  const cfg = config;
  let y = a.yPos;
  const node = (type, name, params, pos) => {
    const id = generateId();
    components.push({ id, type, name, position: pos, params, inputs: [], outputs: [] });
    return id;
  };
  const link = (from, to, fromPort = "bottom", toPort = "top") => connections.push({ id: generateConnId(), from, to, fromPort, toPort });
  const hidden = config.hidden_size || 1152;
  const heads = config.num_attention_heads || Math.max(1, Math.round(hidden / 64));
  const ffDim = config.intermediate_size || hidden * 4;
  const inCh = cfg.in_channels ?? 4;
  const patch = cfg.patch_size ?? 2;
  const ctxDim = cfg.cross_attention_dim ?? cfg.caption_channels ?? 768;
  const hasCrossAttn = (cfg.cross_attention_dim ?? cfg.caption_channels) != null;
  const isMMDiT = !!cfg._mmdit;
  const numSingleLayers = typeof cfg._num_single_layers === "number" ? cfg._num_single_layers : 0;
  const TIME_EMBED_DIM = 256;
  const tIn = node("input", "Timestep", { shape: [TIME_EMBED_DIM] }, { x: xPos + 280, y });
  const tEmb = node("linear", "TimeEmbed", { inFeatures: TIME_EMBED_DIM, outFeatures: hidden }, { x: xPos + 280, y: y + vs });
  link(tIn, tEmb);
  const cIn = node("input", "TextContext", { shape: [1, 77, ctxDim] }, { x: xPos - 280, y });
  const cProj = node("linear", "ContextProj", { inFeatures: ctxDim, outFeatures: hidden }, { x: xPos - 280, y: y + vs });
  link(cIn, cProj);
  y += vs * 2;
  let last = a.lastId;
  if (a.isUNet) {
    const blockCh = cfg.block_out_channels?.length ? cfg.block_out_channels : [320, 640, 1280, 1280];
    const layersPerBlock = cfg.layers_per_block || 2;
    const headsFor = (ch) => {
      for (const hd of [64, 32, 16, 8]) if (ch % hd === 0) return ch / hd;
      return 1;
    };
    const addCond = (attnId) => {
      link(cProj, attnId, "bottom", "right");
      link(tEmb, attnId, "bottom", "left");
    };
    let cur2 = node("conv2d", "ConvIn", { outChannels: blockCh[0], kernelSize: 3, stride: 1, padding: 1 }, { x: xPos, y });
    link(last, cur2);
    last = cur2;
    y += vs;
    const skips = [];
    for (let s = 0; s < blockCh.length; s++) {
      for (let l = 0; l < layersPerBlock; l++) {
        cur2 = node("conv2d", `Down${s + 1}_Conv${l + 1}`, { outChannels: blockCh[s], kernelSize: 3, stride: 1, padding: 1 }, { x: xPos, y });
        link(last, cur2);
        last = cur2;
        y += vs;
        cur2 = node("swish", `Down${s + 1}_Act${l + 1}`, {}, { x: xPos, y });
        link(last, cur2);
        last = cur2;
        y += vs;
      }
      if (s >= 1) {
        cur2 = node("multiHeadAttention", `Down${s + 1}_Attn`, { numHeads: headsFor(blockCh[s]), hiddenDim: blockCh[s] }, { x: xPos - 100, y });
        link(last, cur2);
        addCond(cur2);
        last = cur2;
        y += vs;
      }
      skips.push(last);
      if (s < blockCh.length - 1) {
        cur2 = node("maxpool2d", `Down${s + 1}_Downsample`, { kernelSize: 2, stride: 2, padding: 0 }, { x: xPos, y });
        link(last, cur2);
        last = cur2;
        y += vs;
      }
    }
    const deep = blockCh[blockCh.length - 1];
    cur2 = node("conv2d", "Mid_Conv1", { outChannels: deep, kernelSize: 3, stride: 1, padding: 1 }, { x: xPos, y });
    link(last, cur2);
    last = cur2;
    y += vs;
    cur2 = node("multiHeadAttention", "Mid_Attn", { numHeads: headsFor(deep), hiddenDim: deep }, { x: xPos - 100, y });
    link(last, cur2);
    addCond(cur2);
    last = cur2;
    y += vs;
    cur2 = node("conv2d", "Mid_Conv2", { outChannels: deep, kernelSize: 3, stride: 1, padding: 1 }, { x: xPos, y });
    link(last, cur2);
    last = cur2;
    y += vs;
    for (let s = blockCh.length - 1; s >= 0; s--) {
      if (s < blockCh.length - 1) {
        cur2 = node("upsample", `Up${s + 1}_Upsample`, { scaleFactor: 2 }, { x: xPos, y });
        link(last, cur2);
        last = cur2;
        y += vs;
      }
      cur2 = node("concatenate", `Up${s + 1}_SkipConcat`, { dim: 0 }, { x: xPos, y });
      link(last, cur2, "bottom", "left");
      link(skips[s], cur2, "bottom", "right");
      last = cur2;
      y += vs;
      for (let l = 0; l < layersPerBlock; l++) {
        cur2 = node("conv2d", `Up${s + 1}_Conv${l + 1}`, { outChannels: blockCh[s], kernelSize: 3, stride: 1, padding: 1 }, { x: xPos, y });
        link(last, cur2);
        last = cur2;
        y += vs;
        cur2 = node("swish", `Up${s + 1}_Act${l + 1}`, {}, { x: xPos, y });
        link(last, cur2);
        last = cur2;
        y += vs;
      }
    }
    cur2 = node("conv2d", "ConvOut", { outChannels: inCh, kernelSize: 3, stride: 1, padding: 1 }, { x: xPos, y });
    link(last, cur2);
    last = cur2;
    y += vs;
    return { lastId: last, yPos: y };
  }
  const numLayers = config.num_hidden_layers || 12;
  let cur = node("conv2d", "PatchEmbed", { outChannels: hidden, kernelSize: patch, stride: patch, padding: 0 }, { x: xPos, y });
  link(last, cur);
  last = cur;
  y += vs;
  let textLast = cProj;
  for (let i = 0; i < numLayers; i++) {
    const attn = node("multiHeadAttention", `SelfAttn_${i + 1}`, { numHeads: heads, hiddenDim: hidden }, { x: xPos - 100, y });
    link(last, attn);
    if (i === 0) link(tEmb, attn, "bottom", "left");
    if (i === 0 && !hasCrossAttn) link(cProj, attn, "bottom", "right");
    const add1 = node("add", `Add_${i + 1}_1`, {}, { x: xPos, y: y + 50 });
    link(last, add1, "bottom", "left");
    link(attn, add1, "bottom", "right");
    let blockOut = node("layerNorm", `LayerNorm_${i + 1}_1`, { normalizedShape: hidden }, { x: xPos, y: y + 100 });
    link(add1, blockOut);
    if (hasCrossAttn) {
      const xattn = node("multiHeadAttention", `CrossAttn_${i + 1}`, { numHeads: heads, hiddenDim: hidden }, { x: xPos - 100, y: y + 150 });
      link(blockOut, xattn);
      link(cProj, xattn, "bottom", "right");
      const addx = node("add", `Add_${i + 1}_x`, {}, { x: xPos, y: y + 200 });
      link(blockOut, addx, "bottom", "left");
      link(xattn, addx, "bottom", "right");
      blockOut = node("layerNorm", `LayerNorm_${i + 1}_x`, { normalizedShape: hidden }, { x: xPos, y: y + 250 });
      link(addx, blockOut);
    }
    const ff = node("feedForward", `FFN_${i + 1}`, { hiddenDim: hidden, ffDim }, { x: xPos, y: y + 300 });
    link(blockOut, ff);
    const add2 = node("add", `Add_${i + 1}_2`, {}, { x: xPos, y: y + 350 });
    link(blockOut, add2, "bottom", "left");
    link(ff, add2, "bottom", "right");
    const n2 = node("layerNorm", `LayerNorm_${i + 1}_2`, { normalizedShape: hidden }, { x: xPos, y: y + 400 });
    link(add2, n2);
    if (isMMDiT) {
      const tb = node("transformerBlock", `TextStream_${i + 1}`, { embedDim: hidden, ffDim }, { x: xPos + 240, y: y + 150 });
      link(textLast, tb);
      link(tb, attn);
      textLast = tb;
      const mod = node("linear", `AdaLN_${i + 1}`, { inFeatures: hidden, outFeatures: hidden * 12 }, { x: xPos + 240, y: y + 260 });
      link(tEmb, mod);
      link(mod, attn);
    }
    last = n2;
    y += vs * 4;
  }
  for (let i = 0; i < numSingleLayers; i++) {
    const sb = node("transformerBlock", `SingleBlock_${i + 1}`, { embedDim: hidden, ffDim }, { x: xPos, y });
    link(last, sb);
    last = sb;
    y += vs;
  }
  cur = node("layerNorm", "FinalNorm", { normalizedShape: hidden }, { x: xPos, y });
  link(last, cur);
  last = cur;
  y += vs;
  cur = node("conv2d", "Unpatchify", { outChannels: inCh, kernelSize: 1, stride: 1, padding: 0 }, { x: xPos, y });
  link(last, cur);
  last = cur;
  y += vs;
  return { lastId: last, yPos: y };
}
function buildModalityTower(a) {
  const { kind, mcfg, textHidden, components, connections, generateId, generateConnId, xPos } = a;
  let y = a.yStart;
  const SP = 130;
  const connect = (from, to) => connections.push({ id: generateConnId(), from, to, fromPort: "bottom", toPort: "top" });
  const towerDim = mcfg.hidden_size || mcfg.mm_embed_dim || mcfg.audio_embed_dim || mcfg.d_model || textHidden;
  const cap = kind === "vision" ? "Vision" : "Audio";
  const inId = generateId();
  const inShape = kind === "vision" ? [mcfg.num_channels || 3, mcfg.image_size || 224, mcfg.image_size || 224] : [1, mcfg.num_mel_bins || mcfg.input_features || 128, 1];
  components.push({
    id: inId,
    type: "input",
    name: `${cap} input`,
    position: { x: xPos, y },
    params: { shape: inShape },
    inputShape: inShape,
    inputs: [],
    outputs: []
  });
  y += SP;
  const imgSize = mcfg.image_size || 224;
  const patchSize = mcfg.patch_size || 16;
  let cur = inId;
  if (kind === "vision") {
    const peId = generateId();
    components.push({
      id: peId,
      type: "patchEmbed",
      name: "PatchEmbed",
      // The registry now derives the patch count from the real upstream feature
      // map and only falls back to `imgSize`, but keep imgSize set: it is what
      // the exporter writes into the generated PyTorch.
      position: { x: xPos, y },
      params: { imgSize, patchSize, embedDim: towerDim, inChans: mcfg.num_channels || 3 },
      inputs: [],
      outputs: []
    });
    connect(cur, peId);
    cur = peId;
    y += SP;
    const patches = Math.max(1, Math.floor(imgSize / Math.max(1, patchSize)) ** 2) + 1;
    const vpId = generateId();
    components.push({
      id: vpId,
      type: "learnedPositionalEmbedding",
      name: "Patch_Position_Embedding",
      position: { x: xPos, y },
      params: { maxLen: patches, embedDim: towerDim },
      inputs: [],
      outputs: []
    });
    connect(cur, vpId);
    cur = vpId;
    y += SP;
  } else {
    const feId = generateId();
    components.push({
      id: feId,
      type: "projection",
      name: "Audio feature embed",
      position: { x: xPos, y },
      params: { inDim: mcfg.num_mel_bins || mcfg.input_features || 128, outDim: towerDim },
      inputs: [],
      outputs: []
    });
    connect(cur, feId);
    cur = feId;
    y += SP;
  }
  const nLayers = mcfg.num_hidden_layers || mcfg.num_layers || mcfg.encoder_layers || 0;
  const nHeads = mcfg.num_attention_heads || mcfg.encoder_attention_heads || mcfg.num_heads || 0;
  const ffDim = mcfg.intermediate_size || mcfg.encoder_ffn_dim || mcfg.ffn_dim || 0;
  const hasFullSpec = nLayers > 0 && nHeads > 0 && ffDim > 0;
  if (hasFullSpec) {
    for (let i = 0; i < nLayers; i++) {
      const normId = generateId();
      components.push({
        id: normId,
        type: "layerNorm",
        name: `${cap}_LN_${i + 1}`,
        position: { x: xPos, y },
        params: { normalizedShape: towerDim },
        inputs: [],
        outputs: []
      });
      connect(cur, normId);
      y += SP;
      const attnId = generateId();
      components.push({
        id: attnId,
        type: "multiHeadAttention",
        name: `${cap}_Attn_${i + 1}`,
        position: { x: xPos, y },
        params: { numHeads: nHeads, hiddenDim: towerDim },
        inputs: [],
        outputs: []
      });
      connect(normId, attnId);
      y += SP;
      const addId = generateId();
      components.push({
        id: addId,
        type: "add",
        name: `${cap}_Add_${i + 1}`,
        position: { x: xPos, y },
        params: {},
        inputs: [],
        outputs: []
      });
      connect(attnId, addId);
      connections.push({ id: generateConnId(), from: cur, to: addId, fromPort: "bottom", toPort: "left" });
      y += SP;
      const ffId = generateId();
      components.push({
        id: ffId,
        type: "feedForward",
        name: `${cap}_FFN_${i + 1}`,
        position: { x: xPos, y },
        params: { hiddenDim: towerDim, ffDim },
        inputs: [],
        outputs: []
      });
      connect(addId, ffId);
      cur = ffId;
      y += SP;
    }
  }
  const projId = generateId();
  components.push({
    id: projId,
    type: "projection",
    name: hasFullSpec ? `${cap} projector` : `${cap} encoder (internals not in config)`,
    position: { x: xPos, y },
    params: { inDim: towerDim, outDim: textHidden, projDim: textHidden },
    inputs: [],
    outputs: []
  });
  connect(cur, projId);
  y += SP;
  const tokens = kind === "vision" ? mcfg.num_soft_tokens || Math.floor((imgSize / patchSize) ** 2) || 256 : mcfg.num_soft_tokens || 256;
  const rsId = generateId();
  components.push({
    id: rsId,
    type: "reshape",
    name: `${cap} tokens`,
    position: { x: xPos, y },
    params: { shape: [1, tokens, textHidden] },
    inputs: [],
    outputs: []
  });
  connect(projId, rsId);
  return rsId;
}
function convertHFConfigToModel(modelId, config) {
  const components = [];
  const connections = [];
  const generateId = () => `comp-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  const generateConnId = () => `conn-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  let yPos = 50;
  const xPos = 200;
  const verticalSpacing = 200;
  const architectureClass = config.architectures?.[0] || "";
  const modelType = config.model_type || "";
  const inputId = generateId();
  let inputShape;
  const isDiffusionEarly = !!config._diffusion || /transformer2dmodel|transformer3dmodel|unet2dconditionmodel|unet2dmodel|controlnet|consistency|\bdit\b|mmdit|pixart|\bflux\b|stablediffusion3|sd3transformer|sd35|lumina|\bsana\b|auraflow|hunyuandit|cogview|cogvideo|kandinsky|\bmochi\b|\blatte\b|\bltx\b|ltxvideo|allegro|easyanimate/.test((architectureClass + " " + modelType).toLowerCase());
  const isViTOrCLIP = architectureClass.toLowerCase().includes("vit") || modelType.toLowerCase() === "vit" || architectureClass.toLowerCase().includes("clip") || modelType.toLowerCase().includes("clip") || modelId.toLowerCase().includes("clip");
  if (isDiffusionEarly) {
    const latentCh = config.in_channels ?? 4;
    const latentHW = config.sample_size ?? 64;
    inputShape = [latentCh, latentHW, latentHW];
  } else if (isViTOrCLIP) {
    const imageSize = config.image_size || 224;
    const numChannels = config.num_channels || 3;
    inputShape = [1, numChannels, imageSize, imageSize];
  } else if (architectureClass.toLowerCase().includes("resnet") || modelType.toLowerCase().includes("resnet") || modelType.toLowerCase() === "vgg" || architectureClass.toLowerCase().includes("vgg") || modelType.toLowerCase().includes("mobilenet") || modelType.toLowerCase().includes("efficientnet") || modelType.toLowerCase().includes("convnext") || modelType.toLowerCase().includes("swin")) {
    const imageSize = config.image_size || 224;
    const numChannels = config.num_channels || 3;
    inputShape = [1, numChannels, imageSize, imageSize];
  } else {
    inputShape = [1, config.max_position_embeddings || 512];
  }
  components.push({
    id: inputId,
    type: "input",
    name: isDiffusionEarly ? "Latent" : "Input",
    position: { x: xPos, y: yPos },
    params: { shape: inputShape },
    inputs: [],
    outputs: [],
    inputShape
  });
  yPos += verticalSpacing;
  let lastId = inputId;
  const archLower = architectureClass.toLowerCase();
  const typeLower = modelType.toLowerCase();
  const modelIdLower = modelId.toLowerCase();
  const isCLIP = archLower.includes("clip") || typeLower.includes("clip") || modelIdLower.includes("clip");
  const ENCODER_DECODER_TYPES = ["t5", "mt5", "bart", "mbart", "pegasus", "marian", "whisper", "wav2vec2"];
  const isEncoderDecoder = ENCODER_DECODER_TYPES.some(
    (t) => typeLower.includes(t) || archLower.includes(t) || modelIdLower.includes(t)
  );
  const GPT_LIKE_TYPES = [
    "gpt",
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "gpt-j",
    "codegen",
    "llama",
    "llama2",
    "llama3",
    "opt",
    "bloom",
    "falcon",
    "mistral",
    "mixtral",
    "phi",
    "phi3",
    "qwen",
    "qwen2",
    "gemma",
    "gemma2",
    "deepseek",
    "cohere",
    "olmo",
    "starcoder",
    "codellama",
    "yi",
    "internlm",
    "baichuan",
    "aquila",
    "chatglm",
    "stablelm",
    "mpt",
    "pythia",
    // Decoder-only multimodal LLMs (image/audio tokens fused into the text
    // stream). Their top-level model_type is a wrapper name, so list them here;
    // the nested text decoder's model_type is also matched via _textModelType.
    "llava",
    "paligemma",
    "idefics",
    "mllama",
    "internvl",
    "smolvlm",
    "pixtral",
    "minicpm"
  ];
  const isDiffusion = !!config._diffusion || /transformer2dmodel|transformer3dmodel|unet2dconditionmodel|unet2dmodel|controlnet|consistency|\bdit\b|mmdit|pixart|\bflux\b|stablediffusion3|sd3transformer|sd35|lumina|\bsana\b|auraflow|hunyuandit|cogview|cogvideo|kandinsky|\bmochi\b|\blatte\b|\bltx\b|ltxvideo|allegro|easyanimate/.test(archLower + " " + typeLower);
  const denoiserKind = config._denoiser || (archLower.includes("unet") || typeLower === "unet" || archLower.includes("controlnet") || archLower.includes("consistency") ? "unet" : "transformer");
  const isUNet = isDiffusion && denoiserKind === "unet";
  const textTypeLower = String(config._textModelType ?? "").toLowerCase();
  const isGPTLike = !isCLIP && !isEncoderDecoder && !isDiffusion && (GPT_LIKE_TYPES.some((t) => typeLower.includes(t) || archLower.includes(t) || textTypeLower.includes(t)) || GPT_LIKE_TYPES.some((t) => modelIdLower.includes(t)) || // Fallback: any `*ForCausalLM` architecture is a decoder-only LM even when
  // its family isn't in the keyword list (e.g. MoonshotKimiaForCausalLM). The
  // decoder builder then reads hidden_act (→ SwiGLU) and num_key_value_heads
  // (→ GQA) from the config, so it isn't undercounted by the generic path.
  archLower.endsWith("forcausallm"));
  const GATED_DECODER_STEMS = [
    "llama",
    "mistral",
    "mixtral",
    "qwen",
    "gemma",
    "phi3",
    "phi4",
    "deepseek",
    "cohere",
    "command",
    "starcoder2",
    "olmo",
    "stablelm",
    "yi",
    "baichuan",
    "internlm",
    "minicpm",
    "granite",
    "aquila",
    "chatglm"
  ];
  const actLower = String(config.hidden_act ?? config.hidden_activation ?? "").toLowerCase();
  const isGatedDecoder = GATED_DECODER_STEMS.some((t) => typeLower.includes(t) || modelIdLower.includes(t)) || /silu|swish|geglu|swiglu/.test(actLower);
  const BERT_LIKE_TYPES = [
    "bert",
    "roberta",
    "distilbert",
    "albert",
    "electra",
    "deberta",
    "xlnet",
    "camembert",
    "xlm",
    "flaubert",
    "funnel",
    "layoutlm",
    "mpnet",
    "squeezebert"
  ];
  const isBERTLike = !isCLIP && !isEncoderDecoder && !isGPTLike && !isDiffusion && (BERT_LIKE_TYPES.some((t) => archLower.includes(t) || typeLower.includes(t)) || // Generic encoder: has hidden_size/layers/heads but no decoder indicators
  config.hidden_size && config.num_hidden_layers && config.num_attention_heads && !archLower.includes("vit") && !typeLower.includes("vit") && !archLower.includes("resnet") && !typeLower.includes("resnet"));
  const isConformerCTC = !isDiffusion && (modelIdLower.includes("parakeet") || modelIdLower.includes("conformer") || typeLower.includes("conformer") || archLower.includes("conformer") || (archLower.includes("ctc") || modelIdLower.includes("ctc")) && !modelIdLower.includes("wav2vec") && !typeLower.includes("wav2vec"));
  if (isConformerCTC) {
    const dModel = config.hidden_size ?? config.d_model ?? config.encoder_dim ?? 1024;
    const numLayers = config.num_hidden_layers ?? config.num_layers ?? config.n_layers ?? 24;
    const numHeads = config.num_attention_heads ?? config.n_heads ?? 8;
    const ffDim = config.intermediate_size ?? dModel * 4;
    const vocab = config.vocab_size ?? 1025;
    const melId = generateId();
    components.push({
      id: melId,
      type: "melSpectrogram",
      name: "MelSpectrogram",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: melId, fromPort: "bottom", toPort: "top" });
    lastId = melId;
    yPos += verticalSpacing;
    for (let s = 0; s < 2; s++) {
      const subId = generateId();
      components.push({
        id: subId,
        type: "conv2d",
        name: `Subsample_Conv${s + 1}`,
        position: { x: xPos, y: yPos },
        params: { outChannels: dModel, kernelSize: 3, stride: 2, padding: 1 },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: subId, fromPort: "bottom", toPort: "top" });
      lastId = subId;
      yPos += verticalSpacing;
    }
    const projId = generateId();
    components.push({
      id: projId,
      type: "linear",
      name: "Subsample_Proj",
      position: { x: xPos, y: yPos },
      params: { outFeatures: dModel },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: projId, fromPort: "bottom", toPort: "top" });
    lastId = projId;
    yPos += verticalSpacing;
    const posId = generateId();
    components.push({
      id: posId,
      type: "positionalEncoding",
      name: "Rel_PosEnc",
      position: { x: xPos, y: yPos },
      params: { dModel, embedDim: dModel, maxLen: config.max_source_positions ?? 5e3 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: posId, fromPort: "bottom", toPort: "top" });
    lastId = posId;
    yPos += verticalSpacing;
    for (let i = 0; i < numLayers; i++) {
      const blkId = generateId();
      components.push({
        id: blkId,
        type: "conformerBlock",
        name: `Conformer_${i + 1}`,
        position: { x: xPos, y: yPos },
        params: { dModel, numHeads, ffDim },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: blkId, fromPort: "bottom", toPort: "top" });
      lastId = blkId;
      yPos += verticalSpacing;
    }
    const ctcId = generateId();
    components.push({
      id: ctcId,
      type: "linear",
      name: "CTC_Head",
      position: { x: xPos, y: yPos },
      params: { inFeatures: dModel, outFeatures: vocab },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: ctcId, fromPort: "bottom", toPort: "top" });
    lastId = ctcId;
    yPos += verticalSpacing;
  } else if (isDiffusion) {
    const adv = buildDiffusionDenoiser({
      isUNet,
      config,
      components,
      connections,
      generateId,
      generateConnId,
      lastId,
      xPos,
      yPos,
      verticalSpacing
    });
    lastId = adv.lastId;
    yPos = adv.yPos;
    const vae = config._vae;
    if (vae) {
      const chans = Array.isArray(vae.block_out_channels) && vae.block_out_channels.length ? vae.block_out_channels : [128, 256, 512, 512];
      const lpb = vae.layers_per_block || 2;
      const outC = vae.in_channels || 3;
      const conv = (name, inC, oc, k = 3) => {
        const id = generateId();
        components.push({
          id,
          type: "conv2d",
          name,
          position: { x: xPos, y: yPos },
          params: { inChannels: inC, outChannels: oc, kernelSize: k, stride: 1, padding: Math.floor(k / 2) },
          inputs: [],
          outputs: []
        });
        connections.push({ id: generateConnId(), from: lastId, to: id, fromPort: "bottom", toPort: "top" });
        lastId = id;
        yPos += verticalSpacing;
      };
      const rev = [...chans].reverse();
      let prev = vae.latent_channels || config.in_channels || 4;
      conv("VAE_ConvIn", prev, rev[0]);
      prev = rev[0];
      conv("VAE_Mid1", prev, prev);
      conv("VAE_Mid2", prev, prev);
      rev.forEach((ch, s) => {
        for (let l = 0; l <= lpb; l++) {
          conv(`VAE_Up${s + 1}_Res${l + 1}`, prev, ch);
          prev = ch;
        }
        if (s < rev.length - 1) {
          const upId = generateId();
          components.push({
            id: upId,
            type: "upsample",
            name: `VAE_Up${s + 1}_Upsample`,
            position: { x: xPos, y: yPos },
            params: { scaleFactor: 2 },
            inputs: [],
            outputs: []
          });
          connections.push({ id: generateConnId(), from: lastId, to: upId, fromPort: "bottom", toPort: "top" });
          lastId = upId;
          yPos += verticalSpacing;
        }
      });
      conv("VAE_ConvOut", prev, outC);
    }
  } else if (isBERTLike) {
    const numLayers = config.num_hidden_layers || 12;
    const hiddenSize = config.hidden_size || 768;
    const numHeads = config.num_attention_heads || 12;
    const intermediateSize = config.intermediate_size || config.ffn_dim || 3072;
    const embeddingId = generateId();
    components.push({
      id: embeddingId,
      type: "embedding",
      name: "Embedding",
      position: { x: xPos, y: yPos },
      params: {
        vocabSize: config.vocab_size || 30522,
        embeddingDim: hiddenSize,
        maxSeqLen: config.max_position_embeddings || 512
      },
      inputs: [],
      outputs: []
    });
    connections.push({
      id: generateConnId(),
      from: lastId,
      to: embeddingId,
      fromPort: "bottom",
      toPort: "top"
    });
    lastId = embeddingId;
    yPos += verticalSpacing;
    const bertPosId = generateId();
    components.push({
      id: bertPosId,
      type: "learnedPositionalEmbedding",
      name: "Positional_Embedding",
      position: { x: xPos, y: yPos },
      params: { maxLen: config.max_position_embeddings || 512, embedDim: hiddenSize },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: bertPosId, fromPort: "bottom", toPort: "top" });
    lastId = bertPosId;
    yPos += verticalSpacing;
    const layerSpacing = 450;
    const componentSpacing = 120;
    for (let i = 0; i < numLayers; i++) {
      const layerStartY = yPos;
      const attentionId = generateId();
      components.push({
        id: attentionId,
        type: "multiHeadAttention",
        name: `Attention_${i + 1}`,
        position: { x: xPos - 150, y: layerStartY },
        params: {
          numHeads,
          hiddenDim: hiddenSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: lastId,
        to: attentionId,
        fromPort: "bottom",
        toPort: "top"
      });
      const addId = generateId();
      components.push({
        id: addId,
        type: "add",
        name: `Add_${i + 1}`,
        position: { x: xPos, y: layerStartY + componentSpacing },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: lastId,
        to: addId,
        fromPort: "bottom",
        toPort: "left"
      });
      connections.push({
        id: generateConnId(),
        from: attentionId,
        to: addId,
        fromPort: "bottom",
        toPort: "right"
      });
      const norm1Id = generateId();
      components.push({
        id: norm1Id,
        type: "layerNorm",
        name: `LayerNorm_${i + 1}_1`,
        position: { x: xPos, y: layerStartY + componentSpacing * 2 },
        params: {
          normalizedShape: hiddenSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: addId,
        to: norm1Id,
        fromPort: "bottom",
        toPort: "top"
      });
      const ffId = generateId();
      components.push({
        id: ffId,
        type: "feedForward",
        name: `FFN_${i + 1}`,
        position: { x: xPos, y: layerStartY + componentSpacing * 3 },
        params: {
          hiddenDim: hiddenSize,
          ffDim: intermediateSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: norm1Id,
        to: ffId,
        fromPort: "bottom",
        toPort: "top"
      });
      lastId = ffId;
      yPos = layerStartY + layerSpacing;
    }
  } else if (isGPTLike) {
    const numLayers = config.num_hidden_layers || config.n_layer || 12;
    const hiddenSize = config.hidden_size || config.n_embd || 768;
    const numHeads = config.num_attention_heads || config.n_head || 12;
    const intermediateSize = config.intermediate_size || config.n_inner || config.ffn_dim_hidden || 3072;
    const numKVHeads = config.num_key_value_heads ?? config.num_kv_heads ?? numHeads;
    const headDim = config.head_dim ?? (numHeads > 0 ? Math.floor(hiddenSize / numHeads) : 64);
    const useGQA = isGatedDecoder || config.num_key_value_heads != null;
    const cfgAny = config;
    const useRMSNorm = cfgAny.rms_norm_eps != null || isGatedDecoder && cfgAny.layer_norm_epsilon == null && cfgAny.layer_norm_eps == null;
    const normType = useRMSNorm ? "rmsNorm" : "layerNorm";
    const normLabel = useRMSNorm ? "RMSNorm" : "LayerNorm";
    const useRoPE = useRMSNorm || cfgAny.rope_theta != null || cfgAny.rope_scaling != null;
    const maxPositions = cfgAny.max_position_embeddings ?? cfgAny.n_positions ?? 0;
    const numExperts = config.num_local_experts ?? config.num_experts ?? config.n_routed_experts ?? 0;
    const numSharedExperts = config.n_shared_experts ?? config.num_shared_experts ?? 0;
    const expertDim = config.moe_intermediate_size ?? config.expert_intermediate_size ?? intermediateSize;
    const topK = config.num_experts_per_tok ?? config.experts_per_token ?? config.num_experts_per_token;
    const isMoE = numExperts > 1;
    const firstKDense = config.first_k_dense_replace ?? config.first_k_dense ?? 0;
    const mlpOnlyLayers = Array.isArray(config.mlp_only_layers) ? config.mlp_only_layers : [];
    const sparseStep = config.decoder_sparse_step ?? 0;
    const layerIsMoE = (i) => {
      if (!isMoE) return false;
      if (i < firstKDense) return false;
      if (mlpOnlyLayers.includes(i)) return false;
      if (sparseStep > 1 && (i + 1) % sparseStep !== 0) return false;
      return true;
    };
    const embeddingId = generateId();
    components.push({
      id: embeddingId,
      type: "embedding",
      name: "Embedding",
      position: { x: xPos, y: yPos },
      params: {
        vocabSize: config.vocab_size || 50257,
        embeddingDim: hiddenSize,
        maxSeqLen: config.max_position_embeddings || config.n_positions || 1024
      },
      inputs: [],
      outputs: []
    });
    connections.push({
      id: generateConnId(),
      from: lastId,
      to: embeddingId,
      fromPort: "bottom",
      toPort: "top"
    });
    lastId = embeddingId;
    yPos += verticalSpacing;
    const posId = generateId();
    components.push({
      id: posId,
      type: useRoPE ? "rope" : "learnedPositionalEmbedding",
      name: useRoPE ? "RoPE" : "Positional_Embedding",
      position: { x: xPos, y: yPos },
      params: useRoPE ? { headDim, theta: cfgAny.rope_theta ?? 1e4 } : { maxLen: maxPositions || 1024, embedDim: hiddenSize },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: posId, fromPort: "bottom", toPort: "top" });
    lastId = posId;
    yPos += verticalSpacing;
    const visionCfg = config.vision_config;
    const audioCfg = config.audio_config;
    const towerOuts = [];
    if (visionCfg && typeof visionCfg === "object") {
      towerOuts.push(buildModalityTower({
        kind: "vision",
        mcfg: visionCfg,
        textHidden: hiddenSize,
        components,
        connections,
        generateId,
        generateConnId,
        xPos: xPos - 520,
        yStart: yPos
      }));
    }
    if (audioCfg && typeof audioCfg === "object") {
      towerOuts.push(buildModalityTower({
        kind: "audio",
        mcfg: audioCfg,
        textHidden: hiddenSize,
        components,
        connections,
        generateId,
        generateConnId,
        xPos: xPos + 520,
        yStart: yPos
      }));
    }
    if (towerOuts.length > 0) {
      const fusionId = generateId();
      components.push({
        id: fusionId,
        type: "concatenate",
        name: "Multimodal fusion (concat tokens)",
        position: { x: xPos, y: yPos },
        params: { dim: 1 },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: embeddingId, to: fusionId, fromPort: "bottom", toPort: "top" });
      for (const t of towerOuts) {
        connections.push({ id: generateConnId(), from: t, to: fusionId, fromPort: "bottom", toPort: "left" });
      }
      lastId = fusionId;
      yPos += verticalSpacing;
    }
    const layerSpacing = 600;
    const componentSpacing = 120;
    for (let i = 0; i < numLayers; i++) {
      const layerStartY = yPos;
      const skipId = lastId;
      const norm1Id = generateId();
      components.push({
        id: norm1Id,
        type: normType,
        name: `${normLabel}_${i + 1}_1`,
        position: { x: xPos, y: layerStartY },
        params: { normalizedShape: hiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: norm1Id, fromPort: "bottom", toPort: "top" });
      const attentionId = generateId();
      components.push({
        id: attentionId,
        type: useGQA ? "groupedQueryAttention" : "multiHeadAttention",
        name: `Attention_${i + 1}`,
        position: { x: xPos, y: layerStartY + componentSpacing },
        params: useGQA ? { embedDim: hiddenSize, numHeads, numKVHeads, headDim } : { numHeads, hiddenDim: hiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: norm1Id, to: attentionId, fromPort: "bottom", toPort: "top" });
      const add1Id = generateId();
      components.push({
        id: add1Id,
        type: "add",
        name: `Add_${i + 1}_attn`,
        position: { x: xPos, y: layerStartY + componentSpacing * 2 },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: attentionId, to: add1Id, fromPort: "bottom", toPort: "top" });
      connections.push({ id: generateConnId(), from: skipId, to: add1Id, fromPort: "bottom", toPort: "left" });
      const norm2Id = generateId();
      components.push({
        id: norm2Id,
        type: normType,
        name: `${normLabel}_${i + 1}_2`,
        position: { x: xPos, y: layerStartY + componentSpacing * 3 },
        params: { normalizedShape: hiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: add1Id, to: norm2Id, fromPort: "bottom", toPort: "top" });
      const ffId = generateId();
      const thisMoE = layerIsMoE(i);
      const moeType = numSharedExperts > 0 ? "sharedExpertMoE" : "moeLayer";
      components.push({
        id: ffId,
        type: thisMoE ? moeType : isGatedDecoder ? "swiglu" : "feedForward",
        name: thisMoE ? `MoE_${i + 1}` : `FFN_${i + 1}`,
        position: { x: xPos, y: layerStartY + componentSpacing * 4 },
        // `dim` is what pythonExporter reads for SwiGLU; `embedDim`/`intermediateSize`
        // are what codeGenerator + paramEstimator read. Emit all three so every
        // consumer resolves the real size (not the 512 literal fallback).
        params: thisMoE ? numSharedExperts > 0 ? { embedDim: hiddenSize, numExperts, numSharedExperts, expertDim, topK } : { embedDim: hiddenSize, numExperts, expertDim, topK } : isGatedDecoder ? { embedDim: hiddenSize, dim: hiddenSize, intermediateSize } : { hiddenDim: hiddenSize, ffDim: intermediateSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: norm2Id, to: ffId, fromPort: "bottom", toPort: "top" });
      const add2Id = generateId();
      components.push({
        id: add2Id,
        type: "add",
        name: `Add_${i + 1}_ffn`,
        position: { x: xPos, y: layerStartY + componentSpacing * 5 },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: ffId, to: add2Id, fromPort: "bottom", toPort: "top" });
      connections.push({ id: generateConnId(), from: add1Id, to: add2Id, fromPort: "bottom", toPort: "left" });
      lastId = add2Id;
      yPos = layerStartY + layerSpacing;
    }
    if (config.tie_word_embeddings === false) {
      const finalNormId = generateId();
      components.push({
        id: finalNormId,
        type: normType,
        name: `Final_${normLabel}`,
        position: { x: xPos, y: yPos },
        params: { normalizedShape: hiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: finalNormId, fromPort: "bottom", toPort: "top" });
      yPos += verticalSpacing;
      const lmHeadId = generateId();
      components.push({
        id: lmHeadId,
        type: "linear",
        name: "LM_Head",
        position: { x: xPos, y: yPos },
        params: { inFeatures: hiddenSize, outFeatures: config.vocab_size || 50257, bias: false },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: finalNormId, to: lmHeadId, fromPort: "bottom", toPort: "top" });
      lastId = lmHeadId;
      yPos += verticalSpacing;
    }
  } else if (archLower.includes("vit") || typeLower.includes("vit") || isCLIP || typeLower.includes("swin") || typeLower.includes("deit") || typeLower.includes("beit")) {
    const visionConfig = config.vision_config || config;
    const imageSize = visionConfig.image_size || config.image_size || 224;
    const patchSize = visionConfig.patch_size || config.patch_size || 16;
    const numPatches = Math.floor((imageSize / patchSize) ** 2);
    const hiddenSize = visionConfig.hidden_size || visionConfig.d_model || config.hidden_size || config.d_model || 768;
    const numLayers = visionConfig.num_hidden_layers || visionConfig.num_layers || visionConfig.encoder_layers || config.num_hidden_layers || config.num_layers || 12;
    const numHeads = visionConfig.num_attention_heads || visionConfig.attention_heads || visionConfig.n_head || config.num_attention_heads || config.attention_heads || 12;
    const intermediateSize = visionConfig.intermediate_size || visionConfig.ffn_dim || visionConfig.dim_feedforward || config.intermediate_size || config.ffn_dim || 3072;
    const patchEmbedId = generateId();
    components.push({
      id: patchEmbedId,
      type: "embedding",
      name: "PatchEmbedding",
      position: { x: xPos, y: yPos },
      params: {
        vocabSize: numPatches + 1,
        // +1 for CLS token
        embeddingDim: hiddenSize,
        maxSeqLen: numPatches + 1
      },
      inputs: [],
      outputs: []
    });
    connections.push({
      id: generateConnId(),
      from: lastId,
      to: patchEmbedId,
      fromPort: "bottom",
      toPort: "top"
    });
    lastId = patchEmbedId;
    yPos += verticalSpacing;
    const vitPosId = generateId();
    components.push({
      id: vitPosId,
      type: "learnedPositionalEmbedding",
      name: "Patch_Position_Embedding",
      position: { x: xPos, y: yPos },
      params: { maxLen: numPatches + 1, embedDim: hiddenSize },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: vitPosId, fromPort: "bottom", toPort: "top" });
    lastId = vitPosId;
    yPos += verticalSpacing;
    for (let i = 0; i < numLayers; i++) {
      const attentionId = generateId();
      components.push({
        id: attentionId,
        type: "multiHeadAttention",
        name: `Attention_${i + 1}`,
        position: { x: xPos - 100, y: yPos },
        params: {
          numHeads,
          hiddenDim: hiddenSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: lastId,
        to: attentionId,
        fromPort: "bottom",
        toPort: "top"
      });
      const addId = generateId();
      components.push({
        id: addId,
        type: "add",
        name: `Add_${i + 1}`,
        position: { x: xPos, y: yPos + 50 },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: lastId,
        to: addId,
        fromPort: "bottom",
        toPort: "left"
      });
      connections.push({
        id: generateConnId(),
        from: attentionId,
        to: addId,
        fromPort: "bottom",
        toPort: "right"
      });
      const norm1Id = generateId();
      components.push({
        id: norm1Id,
        type: "layerNorm",
        name: `LayerNorm_${i + 1}_1`,
        position: { x: xPos, y: yPos + 100 },
        params: {
          normalizedShape: hiddenSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: addId,
        to: norm1Id,
        fromPort: "bottom",
        toPort: "top"
      });
      const ffId = generateId();
      components.push({
        id: ffId,
        type: "feedForward",
        name: `FFN_${i + 1}`,
        position: { x: xPos, y: yPos + 250 },
        params: {
          hiddenDim: hiddenSize,
          ffDim: intermediateSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: norm1Id,
        to: ffId,
        fromPort: "bottom",
        toPort: "top"
      });
      const add2Id = generateId();
      components.push({
        id: add2Id,
        type: "add",
        name: `Add_${i + 1}_2`,
        position: { x: xPos, y: yPos + 350 },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: norm1Id,
        to: add2Id,
        fromPort: "bottom",
        toPort: "left"
      });
      connections.push({
        id: generateConnId(),
        from: ffId,
        to: add2Id,
        fromPort: "bottom",
        toPort: "right"
      });
      const norm2Id = generateId();
      components.push({
        id: norm2Id,
        type: "layerNorm",
        name: `LayerNorm_${i + 1}_2`,
        position: { x: xPos, y: yPos + 400 },
        params: {
          normalizedShape: hiddenSize
        },
        inputs: [],
        outputs: []
      });
      connections.push({
        id: generateConnId(),
        from: add2Id,
        to: norm2Id,
        fromPort: "bottom",
        toPort: "top"
      });
      lastId = norm2Id;
      yPos += verticalSpacing * 4;
    }
  } else if (archLower.includes("resnet") || typeLower.includes("resnet")) {
    const rcfg = config;
    const depths = Array.isArray(rcfg.depths) ? rcfg.depths : Array.isArray(rcfg.layers) ? rcfg.layers : [2, 2, 2, 2];
    const isBottleneck = rcfg.layer_type ? String(rcfg.layer_type).toLowerCase() === "bottleneck" : Array.isArray(rcfg.hidden_sizes) ? rcfg.hidden_sizes[0] >= 256 : depths.reduce((a, b) => a + b, 0) > 8;
    const stageChannels = Array.isArray(rcfg.hidden_sizes) && rcfg.hidden_sizes.length >= depths.length ? rcfg.hidden_sizes : isBottleneck ? [256, 512, 1024, 2048] : [64, 128, 256, 512];
    const stemChannels = rcfg.embedding_size ?? 64;
    const stemConvId = generateId();
    components.push({
      id: stemConvId,
      type: "conv2d",
      name: "Stem_Conv",
      position: { x: xPos, y: yPos },
      params: { outChannels: stemChannels, kernelSize: 7, stride: 2, padding: 3 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: stemConvId, fromPort: "bottom", toPort: "top" });
    lastId = stemConvId;
    yPos += verticalSpacing;
    const stemBnId = generateId();
    components.push({
      id: stemBnId,
      type: "batchNorm",
      name: "Stem_BN",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: stemBnId, fromPort: "bottom", toPort: "top" });
    lastId = stemBnId;
    yPos += verticalSpacing;
    const stemReluId = generateId();
    components.push({
      id: stemReluId,
      type: "relu",
      name: "Stem_ReLU",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: stemReluId, fromPort: "bottom", toPort: "top" });
    lastId = stemReluId;
    yPos += verticalSpacing;
    const stemPoolId = generateId();
    components.push({
      id: stemPoolId,
      type: "maxpool2d",
      name: "Stem_MaxPool",
      position: { x: xPos, y: yPos },
      params: { kernelSize: 3, stride: 2, padding: 1 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: stemPoolId, fromPort: "bottom", toPort: "top" });
    lastId = stemPoolId;
    yPos += verticalSpacing;
    const conv = (name, params, from) => {
      const id = generateId();
      components.push({ id, type: "conv2d", name, position: { x: xPos, y: yPos }, params, inputs: [], outputs: [] });
      connections.push({ id: generateConnId(), from, to: id, fromPort: "bottom", toPort: "top" });
      yPos += verticalSpacing;
      return id;
    };
    const simple = (type, name, from, x = xPos) => {
      const id = generateId();
      components.push({ id, type, name, position: { x, y: yPos }, params: {}, inputs: [], outputs: [] });
      connections.push({ id: generateConnId(), from, to: id, fromPort: "bottom", toPort: "top" });
      yPos += verticalSpacing;
      return id;
    };
    let prevOutCh = stemChannels;
    for (let stageIdx = 0; stageIdx < depths.length; stageIdx++) {
      const outCh = stageChannels[stageIdx] ?? stageChannels[stageChannels.length - 1];
      const innerCh = isBottleneck ? Math.max(1, Math.floor(outCh / 4)) : outCh;
      const numBlocks = depths[stageIdx] || 1;
      for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        const tag = `Stage${stageIdx + 1}_Block${blockIdx + 1}`;
        const stride = blockIdx === 0 && stageIdx > 0 ? 2 : 1;
        const blockIn = lastId;
        let mainId;
        if (isBottleneck) {
          mainId = conv(`${tag}_Conv1x1_reduce`, { outChannels: innerCh, kernelSize: 1, stride: 1, padding: 0 }, blockIn);
          mainId = simple("batchNorm", `${tag}_BN1`, mainId);
          mainId = simple("relu", `${tag}_ReLU1`, mainId);
          mainId = conv(`${tag}_Conv3x3`, { outChannels: innerCh, kernelSize: 3, stride, padding: 1 }, mainId);
          mainId = simple("batchNorm", `${tag}_BN2`, mainId);
          mainId = simple("relu", `${tag}_ReLU2`, mainId);
          mainId = conv(`${tag}_Conv1x1_expand`, { outChannels: outCh, kernelSize: 1, stride: 1, padding: 0 }, mainId);
          mainId = simple("batchNorm", `${tag}_BN3`, mainId);
        } else {
          mainId = conv(`${tag}_Conv1`, { outChannels: outCh, kernelSize: 3, stride, padding: 1 }, blockIn);
          mainId = simple("batchNorm", `${tag}_BN1`, mainId);
          mainId = simple("relu", `${tag}_ReLU1`, mainId);
          mainId = conv(`${tag}_Conv2`, { outChannels: outCh, kernelSize: 3, stride: 1, padding: 1 }, mainId);
          mainId = simple("batchNorm", `${tag}_BN2`, mainId);
        }
        let shortcutId = blockIn;
        if (stride !== 1 || prevOutCh !== outCh) {
          const dsY = yPos;
          const dsId = generateId();
          components.push({
            id: dsId,
            type: "conv2d",
            name: `${tag}_Downsample`,
            position: { x: xPos + 260, y: dsY },
            params: { outChannels: outCh, kernelSize: 1, stride, padding: 0 },
            inputs: [],
            outputs: []
          });
          connections.push({ id: generateConnId(), from: blockIn, to: dsId, fromPort: "bottom", toPort: "top" });
          const dsBnId = generateId();
          components.push({
            id: dsBnId,
            type: "batchNorm",
            name: `${tag}_Downsample_BN`,
            position: { x: xPos + 260, y: dsY + verticalSpacing },
            params: {},
            inputs: [],
            outputs: []
          });
          connections.push({ id: generateConnId(), from: dsId, to: dsBnId, fromPort: "bottom", toPort: "top" });
          shortcutId = dsBnId;
        }
        const addId = generateId();
        components.push({
          id: addId,
          type: "add",
          name: `${tag}_Add`,
          position: { x: xPos, y: yPos },
          params: {},
          inputs: [],
          outputs: []
        });
        connections.push({ id: generateConnId(), from: mainId, to: addId, fromPort: "bottom", toPort: "top" });
        connections.push({ id: generateConnId(), from: shortcutId, to: addId, fromPort: "bottom", toPort: "left" });
        yPos += verticalSpacing;
        lastId = simple("relu", `${tag}_ReLU_out`, addId);
        prevOutCh = outCh;
      }
    }
    const gapId = generateId();
    components.push({
      id: gapId,
      type: "globalAvgPool2d",
      name: "GlobalAvgPool",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: gapId, fromPort: "bottom", toPort: "top" });
    lastId = gapId;
    yPos += verticalSpacing;
    const flatId = generateId();
    components.push({
      id: flatId,
      type: "flatten",
      name: "Flatten",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: flatId, fromPort: "bottom", toPort: "top" });
    lastId = flatId;
    yPos += verticalSpacing;
    const numClasses = config.num_labels || config.num_classes || 1e3;
    const fcId = generateId();
    components.push({
      id: fcId,
      type: "linear",
      name: "Classifier",
      position: { x: xPos, y: yPos },
      // The last stage's width, not a hard-coded 512: a bottleneck ResNet ends
      // at 2048, and the wrong number both blocked on shape and lost the 2M
      // parameters the classifier actually carries.
      params: { inFeatures: stageChannels[depths.length - 1] ?? stageChannels[stageChannels.length - 1] ?? 512, outFeatures: numClasses },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: fcId, fromPort: "bottom", toPort: "top" });
    lastId = fcId;
    yPos += verticalSpacing;
  } else if (typeLower === "vgg" || archLower.includes("vgg")) {
    const depths = config.depths || [2, 2, 3, 3, 3];
    const outChannels = config.out_channels || [64, 128, 256, 512, 512];
    const hiddenDim = config.hidden_dim || 4096;
    const numClasses = config.num_labels || config.output_dim || 1e3;
    const useBN = config.batch_norm || false;
    for (let si = 0; si < depths.length; si++) {
      const outCh = outChannels[si] ?? 64;
      const numConvs = depths[si] ?? 2;
      for (let ci = 0; ci < numConvs; ci++) {
        const convId = generateId();
        components.push({
          id: convId,
          type: "conv2d",
          name: `Stage${si + 1}_Conv${ci + 1}`,
          position: { x: xPos, y: yPos },
          params: { outChannels: outCh, kernelSize: 3, stride: 1, padding: 1 },
          inputs: [],
          outputs: []
        });
        connections.push({ id: generateConnId(), from: lastId, to: convId, fromPort: "bottom", toPort: "top" });
        lastId = convId;
        yPos += verticalSpacing;
        if (useBN) {
          const bnId = generateId();
          components.push({
            id: bnId,
            type: "batchNorm",
            name: `Stage${si + 1}_BN${ci + 1}`,
            position: { x: xPos, y: yPos },
            params: {},
            inputs: [],
            outputs: []
          });
          connections.push({ id: generateConnId(), from: lastId, to: bnId, fromPort: "bottom", toPort: "top" });
          lastId = bnId;
          yPos += verticalSpacing;
        }
        const reluId = generateId();
        components.push({
          id: reluId,
          type: "relu",
          name: `Stage${si + 1}_ReLU${ci + 1}`,
          position: { x: xPos, y: yPos },
          params: {},
          inputs: [],
          outputs: []
        });
        connections.push({ id: generateConnId(), from: lastId, to: reluId, fromPort: "bottom", toPort: "top" });
        lastId = reluId;
        yPos += verticalSpacing;
      }
      const poolId = generateId();
      components.push({
        id: poolId,
        type: "maxpool2d",
        name: `Stage${si + 1}_MaxPool`,
        position: { x: xPos, y: yPos },
        params: { kernelSize: 2, stride: 2 },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: poolId, fromPort: "bottom", toPort: "top" });
      lastId = poolId;
      yPos += verticalSpacing;
    }
    const avgPoolId = generateId();
    components.push({
      id: avgPoolId,
      type: "adaptiveAvgPool2d",
      name: "AdaptiveAvgPool",
      position: { x: xPos, y: yPos },
      params: { outputSize: 7 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: avgPoolId, fromPort: "bottom", toPort: "top" });
    lastId = avgPoolId;
    yPos += verticalSpacing;
    const flatId = generateId();
    components.push({
      id: flatId,
      type: "flatten",
      name: "Flatten",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: flatId, fromPort: "bottom", toPort: "top" });
    lastId = flatId;
    yPos += verticalSpacing;
    const lastOutCh = outChannels[outChannels.length - 1] ?? 512;
    for (let fi = 0; fi < 2; fi++) {
      const inF = fi === 0 ? lastOutCh * 7 * 7 : hiddenDim;
      const fcId = generateId();
      components.push({
        id: fcId,
        type: "linear",
        name: `FC${fi + 1}`,
        position: { x: xPos, y: yPos },
        params: { inFeatures: inF, outFeatures: hiddenDim },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: fcId, fromPort: "bottom", toPort: "top" });
      lastId = fcId;
      yPos += verticalSpacing;
      const rId = generateId();
      components.push({
        id: rId,
        type: "relu",
        name: `FC_ReLU${fi + 1}`,
        position: { x: xPos, y: yPos },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: rId, fromPort: "bottom", toPort: "top" });
      lastId = rId;
      yPos += verticalSpacing;
      const drId = generateId();
      components.push({
        id: drId,
        type: "dropout",
        name: `Dropout${fi + 1}`,
        position: { x: xPos, y: yPos },
        params: { dropout: 0.5 },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: drId, fromPort: "bottom", toPort: "top" });
      lastId = drId;
      yPos += verticalSpacing;
    }
    const classId = generateId();
    components.push({
      id: classId,
      type: "linear",
      name: "Classifier",
      position: { x: xPos, y: yPos },
      params: { inFeatures: hiddenDim, outFeatures: numClasses },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: classId, fromPort: "bottom", toPort: "top" });
    lastId = classId;
    yPos += verticalSpacing;
  } else if (isEncoderDecoder && typeLower !== "whisper") {
    const encHiddenSize = config.hidden_size || config.d_model || 768;
    const encNumLayers = config.num_encoder_layers || config.encoder_layers || config.num_hidden_layers || 6;
    const decNumLayers = config.num_decoder_layers || config.decoder_layers || config.num_hidden_layers || 6;
    const numHeads = config.num_attention_heads || config.encoder_attention_heads || 8;
    const ffDim = config.intermediate_size || config.d_ff || config.encoder_ffn_dim || 2048;
    const encX = xPos - 200;
    const decX = xPos + 200;
    const encEmbId = generateId();
    components.push({
      id: encEmbId,
      type: "embedding",
      name: "Encoder_Embedding",
      position: { x: encX, y: yPos },
      params: { vocabSize: config.vocab_size || 32128, embeddingDim: encHiddenSize, maxSeqLen: config.max_position_embeddings || 512 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: encEmbId, fromPort: "bottom", toPort: "top" });
    let encLastId = encEmbId;
    yPos += verticalSpacing;
    const relBuckets = config.relative_attention_num_buckets;
    const encPosId = generateId();
    components.push({
      id: encPosId,
      type: relBuckets != null ? "relativePositionBias" : "learnedPositionalEmbedding",
      name: relBuckets != null ? "Relative_Position_Bias" : "Encoder_Position_Embedding",
      position: { x: encX, y: yPos },
      params: relBuckets != null ? {
        numHeads: config.num_heads ?? config.encoder_attention_heads ?? 8,
        numBuckets: relBuckets,
        maxDistance: config.relative_attention_max_distance ?? 128
      } : { maxLen: config.max_position_embeddings || 1024, embedDim: encHiddenSize },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: encLastId, to: encPosId, fromPort: "bottom", toPort: "top" });
    encLastId = encPosId;
    yPos += verticalSpacing;
    const encLayerSpacing = 350;
    const encCompSpacing = 100;
    for (let i = 0; i < encNumLayers; i++) {
      const ly = yPos + i * encLayerSpacing;
      const eAttnId = generateId();
      components.push({
        id: eAttnId,
        type: "multiHeadAttention",
        name: `Enc_Attn_${i + 1}`,
        position: { x: encX, y: ly },
        params: { numHeads, hiddenDim: encHiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: encLastId, to: eAttnId, fromPort: "bottom", toPort: "top" });
      const eAddId = generateId();
      components.push({
        id: eAddId,
        type: "add",
        name: `Enc_Add_${i + 1}`,
        position: { x: encX, y: ly + encCompSpacing },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: encLastId, to: eAddId, fromPort: "bottom", toPort: "left" });
      connections.push({ id: generateConnId(), from: eAttnId, to: eAddId, fromPort: "bottom", toPort: "right" });
      const eNormId = generateId();
      components.push({
        id: eNormId,
        type: "layerNorm",
        name: `Enc_Norm_${i + 1}`,
        position: { x: encX, y: ly + encCompSpacing * 2 },
        params: { normalizedShape: encHiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: eAddId, to: eNormId, fromPort: "bottom", toPort: "top" });
      const eFfId = generateId();
      components.push({
        id: eFfId,
        type: "feedForward",
        name: `Enc_FFN_${i + 1}`,
        position: { x: encX, y: ly + encCompSpacing * 3 },
        params: { hiddenDim: encHiddenSize, ffDim },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: eNormId, to: eFfId, fromPort: "bottom", toPort: "top" });
      encLastId = eFfId;
    }
    const decEmbId = generateId();
    components.push({
      id: decEmbId,
      type: "embedding",
      name: "Decoder_Embedding",
      position: { x: decX, y: encEmbId ? verticalSpacing : yPos },
      // T5/BART share one token-embedding matrix across encoder and decoder, so the
      // decoder copy is weight-tied (counted on the encoder embedding, not again here).
      params: { vocabSize: config.vocab_size || 32128, embeddingDim: encHiddenSize, maxSeqLen: config.max_position_embeddings || 512, tied: true },
      inputs: [],
      outputs: []
    });
    let decLastId = decEmbId;
    connections.push({ id: generateConnId(), from: lastId, to: decEmbId, fromPort: "bottom", toPort: "top" });
    const decLayerSpacing = 800;
    const decCompSpacing = 100;
    for (let i = 0; i < decNumLayers; i++) {
      const ly = verticalSpacing + i * decLayerSpacing;
      const dSelfAttnId = generateId();
      components.push({
        id: dSelfAttnId,
        type: "multiHeadAttention",
        name: `Dec_SelfAttn_${i + 1}`,
        position: { x: decX, y: ly },
        params: { numHeads, hiddenDim: encHiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: decLastId, to: dSelfAttnId, fromPort: "bottom", toPort: "top" });
      const dAddId = generateId();
      components.push({
        id: dAddId,
        type: "add",
        name: `Dec_Add_${i + 1}`,
        position: { x: decX, y: ly + decCompSpacing },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: decLastId, to: dAddId, fromPort: "bottom", toPort: "left" });
      connections.push({ id: generateConnId(), from: dSelfAttnId, to: dAddId, fromPort: "bottom", toPort: "right" });
      const dNorm1Id = generateId();
      components.push({
        id: dNorm1Id,
        type: "layerNorm",
        name: `Dec_Norm1_${i + 1}`,
        position: { x: decX, y: ly + decCompSpacing * 2 },
        params: { normalizedShape: encHiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: dAddId, to: dNorm1Id, fromPort: "bottom", toPort: "top" });
      const dCrossId = generateId();
      components.push({
        id: dCrossId,
        type: "crossAttention",
        name: `Dec_CrossAttn_${i + 1}`,
        position: { x: decX, y: ly + decCompSpacing * 3 },
        params: { numHeads, hiddenDim: encHiddenSize, embedDim: encHiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: dNorm1Id, to: dCrossId, fromPort: "bottom", toPort: "top" });
      connections.push({ id: generateConnId(), from: encLastId, to: dCrossId, fromPort: "right", toPort: "left" });
      const dAdd2Id = generateId();
      components.push({
        id: dAdd2Id,
        type: "add",
        name: `Dec_Add2_${i + 1}`,
        position: { x: decX, y: ly + decCompSpacing * 4 },
        params: {},
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: dNorm1Id, to: dAdd2Id, fromPort: "bottom", toPort: "left" });
      connections.push({ id: generateConnId(), from: dCrossId, to: dAdd2Id, fromPort: "bottom", toPort: "right" });
      const dNorm2Id = generateId();
      components.push({
        id: dNorm2Id,
        type: "layerNorm",
        name: `Dec_Norm2_${i + 1}`,
        position: { x: decX, y: ly + decCompSpacing * 5 },
        params: { normalizedShape: encHiddenSize },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: dAdd2Id, to: dNorm2Id, fromPort: "bottom", toPort: "top" });
      const dFfId = generateId();
      components.push({
        id: dFfId,
        type: "feedForward",
        name: `Dec_FFN_${i + 1}`,
        position: { x: decX, y: ly + decCompSpacing * 6 },
        params: { hiddenDim: encHiddenSize, ffDim },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: dNorm2Id, to: dFfId, fromPort: "bottom", toPort: "top" });
      decLastId = dFfId;
    }
    const projId = generateId();
    const projY = verticalSpacing + decNumLayers * decLayerSpacing;
    const lmHeadTied = config.tie_word_embeddings !== false;
    components.push({
      id: projId,
      type: "linear",
      name: "LM_Head",
      position: { x: decX, y: projY },
      params: { inFeatures: encHiddenSize, outFeatures: config.vocab_size || 32128, ...lmHeadTied ? { tied: true } : {} },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: decLastId, to: projId, fromPort: "bottom", toPort: "top" });
    lastId = projId;
    yPos = projY + verticalSpacing;
  } else if (isEncoderDecoder && (typeLower.includes("whisper") || archLower.includes("whisper") || modelIdLower.includes("whisper"))) {
    const dModel = config.d_model || 512;
    const encLayers = config.encoder_layers || 6;
    const numHeads = config.encoder_attention_heads || 8;
    const ffDim = config.encoder_ffn_dim || 2048;
    const melId = generateId();
    components.push({
      id: melId,
      type: "melSpectrogram",
      name: "MelSpectrogram",
      position: { x: xPos, y: yPos },
      params: {},
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: melId, fromPort: "bottom", toPort: "top" });
    lastId = melId;
    yPos += verticalSpacing;
    const conv1Id = generateId();
    components.push({
      id: conv1Id,
      type: "conv1d",
      name: "Enc_Conv1",
      position: { x: xPos, y: yPos },
      params: { outChannels: dModel, kernelSize: 3, stride: 1, padding: 1 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: conv1Id, fromPort: "bottom", toPort: "top" });
    lastId = conv1Id;
    yPos += verticalSpacing;
    const conv2Id = generateId();
    components.push({
      id: conv2Id,
      type: "conv1d",
      name: "Enc_Conv2",
      position: { x: xPos, y: yPos },
      params: { outChannels: dModel, kernelSize: 3, stride: 2, padding: 1 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: conv2Id, fromPort: "bottom", toPort: "top" });
    lastId = conv2Id;
    yPos += verticalSpacing;
    const encPermuteId = generateId();
    components.push({
      id: encPermuteId,
      type: "permute",
      name: "Enc_ToTokens",
      position: { x: xPos, y: yPos },
      params: { dims: [0, 2, 1] },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: encPermuteId, fromPort: "bottom", toPort: "top" });
    lastId = encPermuteId;
    yPos += verticalSpacing;
    const posEncId = generateId();
    components.push({
      id: posEncId,
      type: "positionalEncoding",
      name: "Enc_PosEnc",
      position: { x: xPos, y: yPos },
      // maxLen is what the encoder actually positions: Whisper's 30-second
      // window is max_source_positions frames, 1500 for every released size.
      params: { dModel, embedDim: dModel, maxLen: config.max_source_positions ?? 1500 },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: lastId, to: posEncId, fromPort: "bottom", toPort: "top" });
    lastId = posEncId;
    yPos += verticalSpacing;
    const layerSpacing = 350;
    const compSpacing = 100;
    for (let i = 0; i < encLayers; i++) {
      const ly = yPos + i * layerSpacing;
      const attnId = generateId();
      components.push({
        id: attnId,
        type: "multiHeadAttention",
        name: `Enc_Attn_${i + 1}`,
        position: { x: xPos, y: ly },
        params: { numHeads, hiddenDim: dModel },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: lastId, to: attnId, fromPort: "bottom", toPort: "top" });
      const normId = generateId();
      components.push({
        id: normId,
        type: "layerNorm",
        name: `Enc_Norm_${i + 1}`,
        position: { x: xPos, y: ly + compSpacing },
        params: { normalizedShape: dModel },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: attnId, to: normId, fromPort: "bottom", toPort: "top" });
      const ffId = generateId();
      components.push({
        id: ffId,
        type: "feedForward",
        name: `Enc_FFN_${i + 1}`,
        position: { x: xPos, y: ly + compSpacing * 2 },
        params: { hiddenDim: dModel, ffDim },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: normId, to: ffId, fromPort: "bottom", toPort: "top" });
      lastId = ffId;
    }
    yPos += encLayers * layerSpacing + verticalSpacing;
    const encOutId = lastId;
    const decX = xPos + 520;
    const decLayers = config.decoder_layers ?? encLayers;
    const decHeads = config.decoder_attention_heads ?? numHeads;
    const decFfDim = config.decoder_ffn_dim ?? ffDim;
    const maxTarget = config.max_target_positions ?? 448;
    let decY = 50;
    const decInId = generateId();
    components.push({
      id: decInId,
      type: "input",
      name: "decoder_tokens",
      position: { x: decX, y: decY },
      params: { shape: [1, maxTarget] },
      inputs: [],
      outputs: []
    });
    decY += verticalSpacing;
    const decEmbId = generateId();
    components.push({
      id: decEmbId,
      type: "embedding",
      name: "Dec_Token_Embedding",
      position: { x: decX, y: decY },
      params: { vocabSize: config.vocab_size || 51866, embeddingDim: dModel, maxSeqLen: maxTarget },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: decInId, to: decEmbId, fromPort: "bottom", toPort: "top" });
    decY += verticalSpacing;
    const decPosId = generateId();
    components.push({
      id: decPosId,
      type: "learnedPositionalEmbedding",
      name: "Dec_PosEmbedding",
      position: { x: decX, y: decY },
      params: { maxLen: maxTarget, embedDim: dModel },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: decEmbId, to: decPosId, fromPort: "bottom", toPort: "top" });
    decY += verticalSpacing;
    let decLastId = decPosId;
    for (let i = 0; i < decLayers; i++) {
      const ly = decY + i * layerSpacing;
      const selfId = generateId();
      components.push({
        id: selfId,
        type: "causalAttention",
        name: `Dec_SelfAttn_${i + 1}`,
        position: { x: decX, y: ly },
        params: { numHeads: decHeads, embedDim: dModel },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: decLastId, to: selfId, fromPort: "bottom", toPort: "top" });
      const crossId = generateId();
      components.push({
        id: crossId,
        type: "crossAttention",
        name: `Dec_CrossAttn_${i + 1}`,
        position: { x: decX, y: ly + compSpacing },
        params: { numHeads: decHeads, embedDim: dModel, kvDim: dModel },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: selfId, to: crossId, fromPort: "bottom", toPort: "top" });
      connections.push({ id: generateConnId(), from: encOutId, to: crossId, fromPort: "bottom", toPort: "left" });
      const dNormId = generateId();
      components.push({
        id: dNormId,
        type: "layerNorm",
        name: `Dec_Norm_${i + 1}`,
        position: { x: decX, y: ly + compSpacing * 2 },
        params: { normalizedShape: dModel },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: crossId, to: dNormId, fromPort: "bottom", toPort: "top" });
      const dFfId = generateId();
      components.push({
        id: dFfId,
        type: "feedForward",
        name: `Dec_FFN_${i + 1}`,
        position: { x: decX, y: ly + compSpacing * 3 },
        params: { hiddenDim: dModel, ffDim: decFfDim },
        inputs: [],
        outputs: []
      });
      connections.push({ id: generateConnId(), from: dNormId, to: dFfId, fromPort: "bottom", toPort: "top" });
      decLastId = dFfId;
    }
    decY += decLayers * layerSpacing;
    const decFinalNormId = generateId();
    components.push({
      id: decFinalNormId,
      type: "layerNorm",
      name: "Dec_Norm_Final",
      position: { x: decX, y: decY },
      params: { normalizedShape: dModel },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: decLastId, to: decFinalNormId, fromPort: "bottom", toPort: "top" });
    decY += verticalSpacing;
    const lmHeadId = generateId();
    components.push({
      id: lmHeadId,
      type: "linear",
      name: "LM_Head",
      position: { x: decX, y: decY },
      params: { inFeatures: dModel, outFeatures: config.vocab_size || 51866, tied: true },
      inputs: [],
      outputs: []
    });
    connections.push({ id: generateConnId(), from: decFinalNormId, to: lmHeadId, fromPort: "bottom", toPort: "top" });
    lastId = lmHeadId;
    yPos = Math.max(yPos, decY + verticalSpacing);
  } else {
    const hiddenSizeAny = config.hidden_size ?? config.d_model ?? config.n_embd ?? 0;
    const numLayersAny = config.num_hidden_layers ?? config.n_layer ?? config.num_layers ?? 0;
    const numHeadsAny = config.num_attention_heads ?? config.n_head ?? 0;
    console.warn(
      `[hfModelLoader] Unrecognised architecture for ${modelId}: model_type="${modelType}", architectures=${JSON.stringify(config.architectures ?? [])}. Falling back to generic transformer reconstruction (hidden_size=${hiddenSizeAny}, num_layers=${numLayersAny}). If both are 0 the loader will return only an Input node, caller should treat this as an import failure.`
    );
    if (hiddenSizeAny > 0 && numLayersAny > 0) {
      const numLayers = numLayersAny;
      const hiddenSize = hiddenSizeAny;
      const numHeads = numHeadsAny || Math.max(1, Math.round(hiddenSize / 64));
      const intermediateSize = config.intermediate_size ?? config.ffn_dim ?? hiddenSize * 4;
      if (config.vocab_size) {
        const embeddingId = generateId();
        components.push({
          id: embeddingId,
          type: "embedding",
          name: "Embedding",
          position: { x: xPos, y: yPos },
          params: {
            vocabSize: config.vocab_size,
            embeddingDim: hiddenSize,
            maxSeqLen: config.max_position_embeddings || 512
          },
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: lastId,
          to: embeddingId,
          fromPort: "bottom",
          toPort: "top"
        });
        lastId = embeddingId;
        yPos += verticalSpacing;
      }
      const layerSpacing = 500;
      const componentSpacing = 120;
      for (let i = 0; i < numLayers; i++) {
        const layerStartY = yPos;
        const attentionId = generateId();
        components.push({
          id: attentionId,
          type: "multiHeadAttention",
          name: `Attention_${i + 1}`,
          position: { x: xPos - 150, y: layerStartY },
          params: {
            numHeads,
            hiddenDim: hiddenSize
          },
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: lastId,
          to: attentionId,
          fromPort: "bottom",
          toPort: "top"
        });
        const addId = generateId();
        components.push({
          id: addId,
          type: "add",
          name: `Add_${i + 1}`,
          position: { x: xPos, y: layerStartY + componentSpacing },
          params: {},
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: lastId,
          to: addId,
          fromPort: "bottom",
          toPort: "left"
        });
        connections.push({
          id: generateConnId(),
          from: attentionId,
          to: addId,
          fromPort: "bottom",
          toPort: "right"
        });
        const norm1Id = generateId();
        components.push({
          id: norm1Id,
          type: "layerNorm",
          name: `LayerNorm_${i + 1}_1`,
          position: { x: xPos, y: layerStartY + componentSpacing * 2 },
          params: {
            normalizedShape: hiddenSize
          },
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: addId,
          to: norm1Id,
          fromPort: "bottom",
          toPort: "top"
        });
        const ffId = generateId();
        components.push({
          id: ffId,
          type: "feedForward",
          name: `FFN_${i + 1}`,
          position: { x: xPos, y: layerStartY + componentSpacing * 3 },
          params: {
            hiddenDim: hiddenSize,
            ffDim: intermediateSize
          },
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: norm1Id,
          to: ffId,
          fromPort: "bottom",
          toPort: "top"
        });
        const add2Id = generateId();
        components.push({
          id: add2Id,
          type: "add",
          name: `Add_${i + 1}_2`,
          position: { x: xPos, y: layerStartY + componentSpacing * 4 },
          params: {},
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: norm1Id,
          to: add2Id,
          fromPort: "bottom",
          toPort: "left"
        });
        connections.push({
          id: generateConnId(),
          from: ffId,
          to: add2Id,
          fromPort: "bottom",
          toPort: "right"
        });
        const norm2Id = generateId();
        components.push({
          id: norm2Id,
          type: "layerNorm",
          name: `LayerNorm_${i + 1}_2`,
          position: { x: xPos, y: layerStartY + componentSpacing * 5 },
          params: {
            normalizedShape: hiddenSize
          },
          inputs: [],
          outputs: []
        });
        connections.push({
          id: generateConnId(),
          from: add2Id,
          to: norm2Id,
          fromPort: "bottom",
          toPort: "top"
        });
        lastId = norm2Id;
        yPos = layerStartY + layerSpacing;
      }
    }
  }
  const outputId = generateId();
  components.push({
    id: outputId,
    type: "output",
    name: "Output",
    position: { x: xPos, y: yPos },
    params: {},
    inputs: [],
    outputs: []
  });
  connections.push({
    id: generateConnId(),
    from: lastId,
    to: outputId,
    fromPort: "bottom",
    toPort: "top"
  });
  rebuildNodeIO(components, connections);
  components.forEach((comp) => {
    const inferred = inferScopeFromHFName(comp.name);
    if (inferred) comp.scope = inferred;
  });
  return { components, connections };
}
function inferScopeFromHFName(name) {
  if (!name) return void 0;
  let m = /^Enc_(Attn|Add|Norm)(?:_\d+)?_(\d+)$/.exec(name);
  if (m) return `encoder.layer.${parseInt(m[2], 10) - 1}.attention`;
  m = /^Enc_FFN_(\d+)$/.exec(name);
  if (m) return `encoder.layer.${parseInt(m[1], 10) - 1}.ffn`;
  m = /^Dec_(SelfAttn|CrossAttn|Add|Norm1)_(\d+)$/.exec(name);
  if (m) return `decoder.layer.${parseInt(m[2], 10) - 1}.attention`;
  m = /^Dec_FFN_(\d+)$/.exec(name);
  if (m) return `decoder.layer.${parseInt(m[1], 10) - 1}.ffn`;
  m = /^(Attention|Add)_(\d+)$/.exec(name);
  if (m) return `layer.${parseInt(m[2], 10) - 1}.attention`;
  m = /^LayerNorm_(\d+)_1$/.exec(name);
  if (m) return `layer.${parseInt(m[1], 10) - 1}.attention`;
  m = /^Add_(\d+)_2$/.exec(name);
  if (m) return `layer.${parseInt(m[1], 10) - 1}.ffn`;
  m = /^LayerNorm_(\d+)_2$/.exec(name);
  if (m) return `layer.${parseInt(m[1], 10) - 1}.ffn`;
  m = /^FFN_(\d+)$/.exec(name);
  if (m) return `layer.${parseInt(m[1], 10) - 1}.ffn`;
  m = /^Stage(\d+)_Block(\d+)_/.exec(name);
  if (m) return `stage.${parseInt(m[1], 10) - 1}.block.${parseInt(m[2], 10) - 1}`;
  if (/^Stem_/.test(name)) return "stem";
  if (/^Enc_(Conv\d+|PosEnc)$/.test(name)) return "encoder.stem";
  if (/^Encoder_Embedding$/.test(name)) return "encoder.embeddings";
  if (/^Decoder_Embedding$/.test(name)) return "decoder.embeddings";
  if (/^(Embedding|PatchEmbedding|MelSpectrogram)$/.test(name)) return "embeddings";
  if (/^(LM_Head|Classifier|GlobalAvgPool|AdaptiveAvgPool|Flatten)$/.test(name)) return "head";
  return void 0;
}
export {
  ADVISOR_RULE_IDS,
  SHAPE_RULE_IDS,
  convertHFConfigToModel,
  fetchHFModelConfig,
  generatePyTorchCode,
  graphFromPyTorchSource,
  lintModelGraph
};
/*! Bundled license information:

react/cjs/react.production.min.js:
  (**
   * @license React
   * react.production.min.js
   *
   * Copyright (c) Facebook, Inc. and its affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react/cjs/react.development.js:
  (**
   * @license React
   * react.development.js
   *
   * Copyright (c) Facebook, Inc. and its affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

use-sync-external-store/cjs/use-sync-external-store-shim.production.js:
  (**
   * @license React
   * use-sync-external-store-shim.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

use-sync-external-store/cjs/use-sync-external-store-shim.development.js:
  (**
   * @license React
   * use-sync-external-store-shim.development.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

use-sync-external-store/cjs/use-sync-external-store-shim/with-selector.production.js:
  (**
   * @license React
   * use-sync-external-store-shim/with-selector.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

use-sync-external-store/cjs/use-sync-external-store-shim/with-selector.development.js:
  (**
   * @license React
   * use-sync-external-store-shim/with-selector.development.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)
*/
