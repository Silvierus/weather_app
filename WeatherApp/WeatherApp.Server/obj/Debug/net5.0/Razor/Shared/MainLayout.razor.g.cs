#pragma checksum "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Shared\MainLayout.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "41935457372a5bc3d726817bc62f4a162977e4e9"
// <auto-generated/>
#pragma warning disable 1591
namespace WeatherApp.Server.Shared
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Web.Virtualization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using WeatherApp.Server;

#line default
#line hidden
#nullable disable
#nullable restore
#line 10 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using WeatherApp.Server.Shared;

#line default
#line hidden
#nullable disable
    public partial class MainLayout : LayoutComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.OpenElement(0, "div");
            __builder.AddAttribute(1, "class", "page");
            __builder.AddAttribute(2, "b-6r48z8vwok");
            __builder.OpenComponent<Microsoft.AspNetCore.Components.Authorization.AuthorizeView>(3);
            __builder.AddAttribute(4, "ChildContent", (Microsoft.AspNetCore.Components.RenderFragment<Microsoft.AspNetCore.Components.Authorization.AuthenticationState>)((context) => (__builder2) => {
                __builder2.OpenElement(5, "div");
                __builder2.AddAttribute(6, "class", "sidebar");
                __builder2.AddAttribute(7, "b-6r48z8vwok");
                __builder2.OpenComponent<WeatherApp.Server.Shared.NavMenu>(8);
                __builder2.CloseComponent();
                __builder2.CloseElement();
            }
            ));
            __builder.CloseComponent();
            __builder.AddMarkupContent(9, "\r\n        ");
            __builder.OpenElement(10, "div");
            __builder.AddAttribute(11, "class", "main");
            __builder.AddAttribute(12, "b-6r48z8vwok");
            __builder.OpenElement(13, "div");
            __builder.AddAttribute(14, "class", "top-row px-4 auth");
            __builder.AddAttribute(15, "b-6r48z8vwok");
            __builder.OpenComponent<WeatherApp.Server.Shared.LoginDisplay>(16);
            __builder.CloseComponent();
            __builder.CloseElement();
            __builder.AddMarkupContent(17, "\r\n\r\n            ");
            __builder.OpenElement(18, "div");
            __builder.AddAttribute(19, "class", "content px-4");
            __builder.AddAttribute(20, "b-6r48z8vwok");
            __builder.AddContent(21, 
#nullable restore
#line 15 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Shared\MainLayout.razor"
                 Body

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.CloseElement();
            __builder.CloseElement();
        }
        #pragma warning restore 1998
    }
}
#pragma warning restore 1591
